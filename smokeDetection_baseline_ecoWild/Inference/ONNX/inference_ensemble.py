import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import cv2
import time
import Jetson.GPIO as GPIO
import onnxruntime as ort 
import torch
import torch.profiler

# GPIO 설정
output_pin = 33
GPIO.setmode(GPIO.BOARD)
GPIO.setup(output_pin, GPIO.OUT)
GPIO.output(output_pin, GPIO.LOW)



# 사용자 정의 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, image_dir, mode, transform=None):
        self.image_dir = image_dir
        self.mode = mode
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        GPIO.output(output_pin, GPIO.HIGH)  # 핀을 HIGH로 설정
        start_time = time.time()        
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.mode == "whole":
            image = self.process_image(np.array(image), crop_height=1120, resize_dimensions=(1536, 2016))
            image = Image.fromarray(image)
            tiles = None
        elif self.mode == "tiled":
            image = self.process_image(np.array(image), crop_height=1216, resize_dimensions=(1536, 2368))
            tiles = self.process_image_tiled(image, tile_size=640, overlap=64)
            image = tiles

        if self.mode == "tiled" and tiles is not None:
            image = [self.transform(Image.fromarray(tile)) for tile in tiles]
        else:
            image = self.transform(image) if self.transform else image
            tiles = 0
        return image, img_name, tiles, start_time

    def process_image(self, img, crop_height, resize_dimensions):
        img = cv2.resize(img, (resize_dimensions[1], resize_dimensions[0]))
        img = img[resize_dimensions[0] - crop_height:, :]
        return img

    def process_image_tiled(self, img, tile_size, overlap):
        tiles = []
        step_size = tile_size - overlap
        for y in range(0, img.shape[0] - tile_size + 1, step_size):
            for x in range(0, img.shape[1] - tile_size + 1, step_size):
                tile = img[y:y + tile_size, x:x + tile_size]
                tiles.append(tile)
        return tiles

def inference(model, data_loader,  mode="tiled", save_csv=None, model_type="yolo", conf_y=None, conf_r=None):    
    results = []
    inference_times = []
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/'),
        record_shapes=True,
        with_stack=True
    ) as prof:  # 프로파일러 시작
    # with torch.no_grad():
        for inputs_resnet, img_names, tiles , start_time in tqdm(data_loader):
            start_time = start_time.numpy()
            resnet_model, yolo_model = model
            preds, inference_time = ensemble_predict_onnx(inputs_resnet, resnet_model, yolo_model, mode, conf_y, conf_r, start_time)

            results.append({"Image": img_names[0], **preds, "Inference_Time": inference_time})
            prof.step()

    if save_csv:
        df_results = pd.DataFrame(results)
        df_results.to_csv(save_csv, index=False)

    return results



# 앙상블 추론 함수 - Smoke/No_Smoke 레이블 추가 (ONNX 지원)
def ensemble_predict_onnx(inputs_resnet, resnet_model, yolo_model, mode, conf_y, conf_r, start_time):
    results = {}
    if mode == "tiled":
        for tile_idx, input_tile in enumerate(inputs_resnet):
            input_tile = input_tile[0].unsqueeze(0).numpy()  # ONNX 추론에 맞게 NumPy 배열로 변환


            yolo_output = yolo_model.run(None, {yolo_model.get_inputs()[0].name: input_tile})[0]

            resnet_output = resnet_model.run(None, {resnet_model.get_inputs()[0].name: input_tile})[0]

            resnet_probs =  1 / (1 + np.exp(-(resnet_output[:, 1])))
            yolo_probs = yolo_output[:, 1]
            yolo_preds = 1 if yolo_probs > conf_y else 0
            ensemble_label = "Smoke" if resnet_probs >= conf_r or yolo_preds == 1 else "No_Smoke"
            if tile_idx == 7:
                inference_time = time.time() - start_time
                GPIO.output(output_pin, GPIO.LOW) # pre fin            
            
            results[f"Ensemble_Tile_{tile_idx}"] = ensemble_label        
    
    else:
        input_resnet = inputs_resnet.numpy()
        yolo_output = yolo_model.run(None, {yolo_model.get_inputs()[0].name: inputs_resnet.unsqueeze(0).numpy().squeeze(0)})[0]
        resnet_output = resnet_model.run(None, {resnet_model.get_inputs()[0].name: input_resnet})[0]

        resnet_probs = 1 / (1 + np.exp(-(resnet_output[:, 1])))
        yolo_probs = yolo_output[:, 1]
        yolo_preds = 1 if yolo_probs > conf_y else 0
        ensemble_label = "Smoke" if resnet_probs >= conf_r or yolo_preds == 1 else "No_Smoke"
        inference_time = time.time() - start_time
        GPIO.output(output_pin, GPIO.LOW) # pre fin  
        results["Ensemble"] = ensemble_label

    return results, inference_time[0]

if __name__ == "__main__":
    mode = "whole"
    imgsz = 640
    resnet_path = "./Model/resnet34_model.onnx"
    # yolo_path = "./Model/yolov8l_cls_whole_golden_best.onnx"
    yolo_path = "./Model/yolov8l-cls_whole_224_best_new.onnx"
    csv = f"Ensemble_{mode}_{imgsz}_onnx.csv"


    if imgsz == 640:
        # Transform 설정
        transform_resnet = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        transform_yolo = transforms.Compose([
            transforms.Resize((640, 640)),  # YOLO는 640
            transforms.ToTensor(),
        ])

    else:
        transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
        ])


    # Dataset 설정
    test_image_dir = './Dataset/20180728_FIRE_rm-w-mobo-c'
    test_dataset = CustomDataset(test_image_dir, mode=mode, transform_resnet=transform_resnet, transform_yolo=transform_yolo)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    resnet = ort.InferenceSession(resnet_path, providers=providers)
    yolo_model = ort.InferenceSession(yolo_path, providers=providers)
    results = inference((resnet, yolo_model), test_loader,  mode=mode, save_csv=csv, model_type="ensemble", conf_y=0.29, conf_r=0.25)
    results = inference((resnet, yolo_model), test_loader,  mode=mode, save_csv=csv, model_type="ensemble", conf_y=0.29, conf_r=0.25)
    print("Measure!!")
    time.sleep(2)
    print("Measure!!")
    results = inference((resnet, yolo_model), test_loader,  mode=mode, save_csv=csv, model_type="ensemble", conf_y=0.29, conf_r=0.25)


    print("Inference completed and results saved.")
