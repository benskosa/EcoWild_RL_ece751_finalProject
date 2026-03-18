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
# GPIO 설정
output_pin = 33
GPIO.setmode(GPIO.BOARD)
GPIO.setup(output_pin, GPIO.OUT)
GPIO.output(output_pin, GPIO.LOW)


# 사용자 정의 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, image_dir, mode,  transform_yolo=None):
        self.image_dir = image_dir
        self.mode = mode
        self.transform_yolo = transform_yolo
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
            image_yolo = [self.transform_yolo(Image.fromarray(tile)) for tile in tiles]
        else:
            image_yolo = self.transform_yolo(image) if self.transform_yolo else image
            tiles = 0
        return image_yolo, img_name, tiles, start_time

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
    for inputs_yolo, img_names, tiles , start_time in tqdm(data_loader):
            start_time = start_time.numpy()
            preds, inference_time= yolo_predict_onnx(inputs_yolo, model, mode, conf_y, start_time)

            results.append({"Image": img_names[0], **preds, "Inference_Time": inference_time})

    if save_csv:
        df_results = pd.DataFrame(results)
        df_results.to_csv(save_csv, index=False)

    return results


# YOLO ONNX 추론 함수 - Smoke/No_Smoke 레이블 추가
def yolo_predict_onnx(inputs_yolo, model, mode, conf_y, start_time):
    results = {}
    if mode == "tiled":
        for tile_idx, input_yolo_tile in enumerate(inputs_yolo):
            yolo_input = input_yolo_tile[0].unsqueeze(0).numpy()  # ONNX는 NumPy 배열을 사용
            if len(yolo_input.shape) == 5:
                yolo_input = yolo_input.squeeze(0)
            yolo_output = model.run(None, {model.get_inputs()[0].name: yolo_input})[0]
            yolo_probs = yolo_output[:, 1]  # NumPy 배열 그대로 사용
            yolo_preds = 1 if yolo_probs > conf_y else 0
            label = "Smoke" if yolo_preds == 1 else "No_Smoke"
            if tile_idx == 7:
                inference_time = time.time() - start_time
                GPIO.output(output_pin, GPIO.LOW) # pre fin

            results[f"YOLO_Tile_{tile_idx}"] = label
    else:
        yolo_input = inputs_yolo.unsqueeze( 0).numpy()
        if len(yolo_input.shape) == 5:
                yolo_input = yolo_input.squeeze(0)
        yolo_output = model.run(None, {model.get_inputs()[0].name: yolo_input})[0]
        yolo_probs = yolo_output[:, 1]  # NumPy 배열 그대로 사용
        yolo_preds = 1 if yolo_probs > conf_y else 0
        label = "Smoke" if yolo_preds == 1 else "No_Smoke"
        test = time.time()
        inference_time = test - start_time
        GPIO.output(output_pin, GPIO.LOW) # pre fin
        results["YOLO"] = label
    return results, inference_time[0]


if __name__ == "__main__":    
    imgsz = 224 
    test_image_dir = './Dataset/20180728_FIRE_rm-w-mobo-c'
    mode = "tiled"
    csv = f"./Yolo_{mode}_{imgsz}_onnx.csv "
    
    # Transform 설정

    transform_yolo = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),  # YOLO는 640으로
        transforms.ToTensor(),
    ])

    # Dataset 설정
    test_dataset = CustomDataset(test_image_dir, mode=mode,  transform_yolo=transform_yolo)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    yolo_model = ort.InferenceSession("./Model/yolov8l-cls_whole_224_best_new.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # yolo_model = ort.InferenceSession("./Model/yolov8l_cls_whole_golden_best.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    results = inference(yolo_model, test_loader, mode=mode, save_csv=csv, model_type="yolo", conf_y=0.25)
    results = inference(yolo_model, test_loader, mode=mode, save_csv=csv, model_type="yolo", conf_y=0.1)
    print("Measure!!")
    time.sleep(2)
    print("Measure!!")
    results = inference(yolo_model, test_loader, mode=mode, save_csv=csv, model_type="yolo", conf_y=0.1)

    print("Inference completed and results saved.")
