import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torchvision.models as models
from ultralytics import YOLO
import pandas as pd
import cv2  # OpenCV 사용을 위해 추가
import time

# 사용자 정의 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, image_dir, mode="whole", transform_resnet=None, transform_yolo=None):
        self.image_dir = image_dir
        self.mode = mode
        self.transform_resnet = transform_resnet
        self.transform_yolo = transform_yolo
        self.image_files = sorted(os.listdir(image_dir))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # mode에 따른 전처리
        if self.mode == "whole":
            # 이미지 리사이징 및 크롭핑
            image = self.process_image_and_labels(np.array(image), crop_height=1120, resize_dimensions=(1536, 2016))
            image = Image.fromarray(image)
            tiles = None
        elif self.mode == "tiled":
            # 이미지 리사이징, 크롭핑 및 타일 분할
            image = self.process_image_and_labels(np.array(image), crop_height=1216, resize_dimensions=(1536, 2368))
            tiles = self.process_image_tiled(image, tile_size=640, overlap=64)
            image = tiles  # 첫 번째 타일만 사용 (필요한 경우 모든 타일 사용)

        # resnet과 yolo에 대한 변환을 개별적으로 적용
        if self.mode == "tiled" and tiles is not None:
            image_resnet = [self.transform_resnet(Image.fromarray(tile)) if self.transform_resnet else Image.fromarray(tile) for tile in tiles]
            image_yolo = [self.transform_yolo(Image.fromarray(tile)) if self.transform_yolo else Image.fromarray(tile) for tile in tiles]
            
        else:
            image_resnet = self.transform_resnet(image) if self.transform_resnet else image
            image_yolo = self.transform_yolo(image) if self.transform_yolo else image
            tiles = 0
        
        return image_resnet, image_yolo, img_name, tiles
    
    def process_image_and_labels(self, img, crop_height, resize_dimensions):
        original_height, original_width = img.shape[:2]

        # 이미지 리사이징
        img = cv2.resize(img, (resize_dimensions[1], resize_dimensions[0]))

        # 이미지 크롭핑
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

# 데이터 전처리 및 증강 설정
transform_resnet = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet에 최적화된 입력 크기
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_yolo = transforms.Compose([
    transforms.Resize((640, 640)),  # YOLO에 최적화된 입력 크기
    transforms.ToTensor(),
])

# 데이터셋 경로 설정 (사용자 데이터셋 경로로 교체)
test_image_dir = './Dataset20180728_FIRE_rm-w-mobo-c'

# 데이터셋 로드
test_dataset = CustomDataset(test_image_dir, mode="whole", transform_resnet=transform_resnet, transform_yolo=transform_yolo)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ResNet 모델 로드 및 최종 레이어 수정 (이진 분류)
resnet = models.resnet34()
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 2)
resnet.load_state_dict(torch.load('./Model/best_resnet34_model_epoch_3.pth'))
resnet.eval()

# YOLOv8 모델 로드
yolo_model = YOLO('./Model/yolov8l_cls_whole_golden_best.pt')

# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet.to(device)

# 앙상블 추론 함수 정의
def ensemble_inference(resnet_model, yolo_model, data_loader, device, mode="whole", save_dir = None):
    resnet_model.eval()
    yolo_model.to(device)
    
    results = []
    last_image = None
    last_tiles = None
    with torch.no_grad():
        for inputs_resnet, inputs_yolo, img_names, tiles in tqdm(data_loader):
            if mode == "tiled":
                # 각 타일에 대해 예측 수행
                for tile_idx, (input_resnet_tile, input_yolo_tile) in enumerate(zip(inputs_resnet, inputs_yolo)):
                    input_resnet_tile = input_resnet_tile.to(device)
                    
                    # ResNet 예측
                    resnet_outputs = resnet_model(input_resnet_tile)
                    resnet_probs = torch.sigmoid(resnet_outputs[:, 1]).item()  # 1일 확률
                    
                    # YOLOv8 예측
                    yolo_image = transforms.ToPILImage()(input_yolo_tile.squeeze(0))
                    yolo_results = yolo_model(yolo_image, conf=0.25)
                    yolo_preds = 1 if yolo_results[0].probs.data.tolist()[1] > 0.25 else 0
                    
                    # 조건에 따라 결과 결정
                    if resnet_probs >= 0.2 and yolo_preds == 1:
                        ensemble_label = "Smoke"
                    else:
                        ensemble_label = "No_Smoke"
                        
                    # 결과 저장
                    results.append({"Image": img_names[0], "Tile_Index": tile_idx, "ResNet34_Prob": resnet_probs, "YOLO_Pred": yolo_preds, "Ensemble_Label": ensemble_label})
                    
                last_tiles = tiles
            else:
                inputs_resnet = inputs_resnet.to(device)
                
                # ResNet 예측
                resnet_outputs = resnet_model(inputs_resnet)
                resnet_probs = torch.sigmoid(resnet_outputs[:, 1]).item()  # 1일 확률
                
                # YOLOv8 예측을 위해 텐서를 이미지로 변환
                yolo_image = transforms.ToPILImage()(inputs_yolo.squeeze(0))  # YOLO 모델에 맞는 이미지 형식으로 변환
                yolo_results = yolo_model(yolo_image, conf=0.25)
                yolo_preds = 1 if yolo_results[0].probs.data.tolist()[1] >0.25 else 0
                
                # 조건에 따라 결과 결정
                if resnet_probs >= 0.2 or yolo_preds == 1:
                    ensemble_label = "Smoke"
                else:
                    ensemble_label = "No_Smoke"
                    
                # 결과 저장
                results.append({"Image": img_names[0], "ResNet34_Prob": resnet_probs, "YOLO_Pred": yolo_preds, "Ensemble_Label": ensemble_label})
                last_image = inputs_yolo
    
    if save_dir:
        if mode == "whole":
            last_image_pil = transforms.ToPILImage()(last_image.squeeze(0))
            last_image_pil.save(os.path.join(save_dir, "last_image.png"))
        elif mode == "tiled" and last_tiles:
            for i, tile in enumerate(last_tiles):
                tile_np = tile.squeeze(0).numpy().astype(np.uint8)  # Tensor를 numpy 배열로 변환하고 형식을 uint8로 변경
                tile_pil = Image.fromarray(tile_np)
                tile_pil.save(os.path.join(save_dir, f"last_tile_{i}.png"))
            print("SAVE tiles")

    return results

def calculate_average_fps(inference_times, mode="whole", num_images=81, num_tiles=8):
    if mode == "whole":
        mean_inference_time = np.sum(inference_times) / num_images
        fps = 1 / mean_inference_time
    elif mode == "tiled":
        mean_inference_time = np.sum(inference_times) / (num_images * num_tiles)
        image_processing_time = mean_inference_time * num_tiles
        fps = 1 / image_processing_time
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return fps

# 앙상블 추론 수행
times = time.time()
results = ensemble_inference(resnet, yolo_model, test_loader, device, mode="whole", save_dir ="/home/elab02/wildfire/img/DATA/tmp" )
retime = time.time()
print(times - retime)
df_results = pd.DataFrame(results)
df_results.to_csv('inference_results_pth_whole.csv', index=False)
print("Completed")
