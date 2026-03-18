import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import cv2
import Jetson.GPIO as GPIO
from tqdm import tqdm
import pandas as pd
import time
import onnxruntime as ort


 # GPIO 설정
output_pin = 33
GPIO.setmode(GPIO.BOARD)
GPIO.setup(output_pin, GPIO.OUT)
GPIO.output(output_pin, GPIO.LOW)

class CustomDataset(Dataset):
    def __init__(self, image_dir, mode, transform_resnet=None):
        self.image_dir = image_dir
        self.mode = mode
        self.transform_resnet = transform_resnet
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
            image_resnet = [self.transform_resnet(Image.fromarray(tile)) for tile in tiles]
        else:
            image_resnet = self.transform_resnet(image) if self.transform_resnet else image
            tiles = 0
        return image_resnet, img_name, tiles, start_time

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


def inference(model, data_loader, mode="tiled", save_csv=None, model_type="yolo", conf_r=None):    
    results = []
    for inputs_resnet, img_names, tiles, start_time in tqdm(data_loader):
        start_time = start_time.numpy()

        inputs_resnet = [input_resnet.cuda() for input_resnet in inputs_resnet]  # GPU로 데이터 전송
        preds, inference_time = resnet_predict_onnx(inputs_resnet, model, mode, conf_r, start_time)

        results.append({"Image": img_names[0], **preds, "Inference_Time": inference_time})

    if save_csv:
        df_results = pd.DataFrame(results)
        df_results.to_csv(save_csv, index=False)

    return results

# ResNet ONNX 추론 함수 - Smoke/No_Smoke 레이블 추가
def resnet_predict_onnx(inputs_resnet, model, mode, conf_r, start_time):
    results = {}
    if mode == "tiled":
        for tile_idx, input_resnet_tile in enumerate(inputs_resnet):
            input_resnet_tile = input_resnet_tile[0].unsqueeze(0).cpu().numpy()  # GPU에서 CPU로 변환 후 NumPy 배열로 변환
            resnet_output = model.run(None, {model.get_inputs()[0].name: input_resnet_tile})[0]
            resnet_probs = 1 / (1 + np.exp(-(resnet_output[:, 1]))) 
            label = "Smoke" if resnet_probs >= conf_r else "No_Smoke"
            if tile_idx == 7:
                inference_time = time.time() - start_time
                GPIO.output(output_pin, GPIO.LOW) # pre fin    
            results[f"ResNet_Tile_{tile_idx}"] = label
    else:
        input_resnet = inputs_resnet[0].unsqueeze(0).cpu().numpy()  # 리스트에서 텐서 꺼낸 후 변환
        resnet_output = model.run(None, {model.get_inputs()[0].name: input_resnet})[0]
        resnet_probs = 1 / (1 + np.exp(-(resnet_output[:, 1]))) 
        label = "Smoke" if resnet_probs >= conf_r else "No_Smoke"
        inference_time = time.time() - start_time
        GPIO.output(output_pin, GPIO.LOW) # post fin

        results["ResNet"] = label

    return results, inference_time[0]


if __name__ == "__main__":
    mode = "tiled"
    imgsz = 224
    csv = f"ResNet_{mode}_{imgsz}_onnx.csv"
    # Transform 설정
    transform_resnet = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Dataset 설정
    test_image_dir = './Dataset/20180728_FIRE_rm-w-mobo-c'
    test_dataset = CustomDataset(test_image_dir, mode=mode, transform_resnet=transform_resnet)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # ONNX 모델 로드 및 GPU 설정
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    resnet = ort.InferenceSession("./Model/resnet34_model.onnx", providers=providers)

    # Inference 실행
    results = inference(resnet, test_loader, mode=mode, save_csv=csv, model_type="resnet", conf_r=0.1)
    results = inference(resnet, test_loader, mode=mode, save_csv=csv, model_type="resnet", conf_r=0.1)
    print("Measure!!")
    time.sleep(2)
    print("Measure!!")
    results = inference(resnet, test_loader, mode=mode, save_csv=csv, model_type="resnet", conf_r=0.1)

    print("Inference completed and results saved.")

