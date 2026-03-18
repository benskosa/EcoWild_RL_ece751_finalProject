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
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

## Test at Jetson Orin nano
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


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TrtModel:
    
    def __init__(self,engine_path,max_batch_size=1,dtype=np.float32):
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
       
    def __call__(self, x: np.ndarray, batch_size=1):
        x = x.astype(self.dtype)
        np.copyto(self.inputs[0].host, x.ravel())
        
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
        
        self.stream.synchronize()
        return [out.host.reshape(batch_size, -1) for out in self.outputs]





def inference(model, data_loader, mode="tiled", save_csv=None, model_type="resnet", conf_r=None):
    results = []
    for inputs_resnet, img_names, tiles, start_time in tqdm(data_loader):
        inputs_resnet = [input_resnet.cuda() for input_resnet in inputs_resnet]  # GPU로 데이터 전송
        preds, inference_time = resnet_predict_tensorrt(inputs_resnet, model, mode, conf_r, start_time)

        results.append({"Image": img_names[0], **preds, "Inference_Time": inference_time})

    if save_csv:
        df_results = pd.DataFrame(results)
        df_results.to_csv(save_csv, index=False)

    return results


# ResNet TensorRT 추론 함수 - Smoke/No_Smoke 레이블 추가
def resnet_predict_tensorrt(inputs_resnet, model, mode, conf_r, start_time):
    results = {}

    if mode == "tiled":
        for tile_idx, input_resnet_tile in enumerate(inputs_resnet):
            input_resnet_tile = input_resnet_tile[0].unsqueeze(0).cpu().numpy()  # GPU에서 CPU로 변환 후 NumPy 배열로 변환

            # Inference using TrtModel
            resnet_output = model(input_resnet_tile)
            print(np.array(resnet_output).shape)
            print(len(resnet_output))
            resnet_probs = 1 / (1 + np.exp(-(resnet_output[0][0][1])))  # Apply sigmoid on second logit
            label = "Smoke" if resnet_probs >= conf_r else "No_Smoke"
            if tile_idx == 7:
                inference_time = time.time() - start_time
                GPIO.output(output_pin, GPIO.LOW)  # Set pin LOW when done
            results[f"ResNet_Tile_{tile_idx}"] = label
    else:
        input_resnet = inputs_resnet[0].unsqueeze(0).cpu().numpy()  # 리스트에서 텐서 꺼낸 후 변환

        # Inference using TrtModel
        resnet_output = model(input_resnet)
        resnet_probs = 1 / (1 + np.exp(-(resnet_output[0][0][1])))  # Apply sigmoid on second logit
        label = "Smoke" if resnet_probs >= conf_r else "No_Smoke"
        inference_time = time.time() - start_time
        GPIO.output(output_pin, GPIO.LOW)  # Set pin LOW when done
        results["ResNet"] = label

    return results, inference_time


if __name__ == "__main__":
    mode = "whole"
    imgsz = 224
    csv = f"ResNet_{mode}_{imgsz}_tensorrt.csv"

    # Transform 설정
    transform_resnet = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Dataset 설정
    test_image_dir = './Dataset/20180728_FIRE_rm-w-mobo-c'
    test_dataset = CustomDataset(test_image_dir, mode=mode, transform_resnet=transform_resnet)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # TensorRT 모델 로드 및 GPU 설정
    model = TrtModel("./Model/resnet34_model.trt")

    # Inference 실행
    results = inference(model, test_loader, mode=mode, save_csv=csv, model_type="resnet", conf_r=0.1)
    results = inference(model, test_loader, mode=mode, save_csv=csv, model_type="resnet", conf_r=0.1)
    print("Measure!!")
    time.sleep(2)
    print("Measure!!")
    results = inference(model, test_loader, mode=mode, save_csv=csv, model_type="resnet", conf_r=0.1)

    print("Inference completed and results saved.")