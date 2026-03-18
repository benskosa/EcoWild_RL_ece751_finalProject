from ultralytics import YOLO
import argparse
from datetime import datetime
import os, yaml



def build_yaml(data_path):
    print("data path: ",data_path)
    if data_path.split("/")[-1] == "":
        data_name = data_path.split("/")[-2]
    else: 
        data_name = data_path.split("/")[-1]
    print("data name:", data_name)
    yaml_path = f'{data_path}/{data_name}.yaml'
    print("yaml path:", yaml_path)
    if mode == "detect": 
        # yaml file만들기
        data = {
            'train': f'{data_path}/train/images/',
            'val': f'{data_path}/val/images/',
            'test': f'{data_path}/test/images/',
            'names': ['smoke'],
            'nc': 1
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f)

        with open(yaml_path, 'r') as f:
            wildfire_yaml = yaml.safe_load(f)
    return yaml_path

def save_configuration(src_path, model, mode, data_path, epochs, imgsz, device, resume, batch, save, patience, workers, name, optimizer, plots, train_fire_txt):
            # write setting
    with open(f"{src_path}/runs/{mode}/{name}/training_conf.txt", 'w') as f:
        f.write(f"model={model} \n")
        f.write(f"mode = {mode} \n")
        f.write(f"data_path = {data_path} \n")
        f.write(f"epochs = {epochs} \n")
        f.write(f"imgsz = {imgsz} \n")
        f.write(f"device = {device} \n")
        f.write(f"resume = {resume} \n")
        f.write(f"batch = {batch} \n")
        f.write(f"save = {save} \n")
        f.write(f"patience = {patience} \n")
        f.write(f"workers = {workers} \n")
        f.write(f"name = {name} \n")
        f.write(f"optimizer = {optimizer} \n")
        f.write(f"plots = {plots} \n")
        f.write(f"train_fire_txt = {train_fire_txt} \n")
        #f.write(f"split_ratio = {split_ratio} \n")

def detect_train_yolo(model, yaml_path, epochs, batch, imgsz, device, resume, save, patience, workers, name, optimizer, plots):
    model = YOLO(model)
    print("START TRAINING")
    results = model.train(data=yaml_path, epochs=epochs, batch=batch, imgsz=imgsz, device=device, resume=resume,
    save=save, patience=patience, workers=workers, name=name, optimizer=optimizer, plots=plots)

def cls_train_yolo(model, data_path, epochs, batch, imgsz, device, resume, save, patience, workers, name, optimizer, plots):
    model = YOLO(model)
    print("START TRAINING")
    results = model.train(data=data_path, epochs=epochs, batch=batch, imgsz=imgsz, device=device, resume=resume,
    save=save, patience=patience, workers=workers, name=name, optimizer=optimizer, plots=plots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "YOLOv8 training.py for WILDFIRE")

    # Define arguments
    #parser.add_argument('--option', action = '', help='', type=int)
    parser.add_argument('-p', '--model', help='type yolo model', type=str, default=None)
    parser.add_argument('-m', '--mode', help = 'type mode (detect or cls)', type=str, default=None)
    parser.add_argument('-a', '--data_path', help = 'type path of dataset', type=str) # /home/gkim283/wildfire_detect/data/GJ_DATA/test_data
    parser.add_argument('-e', '--epochs', type=int, default=1000)
    parser.add_argument('-i', '--imgsz', type=int, default=640)
    parser.add_argument('-d', '--device', nargs='+', type=int, help="List of device indices", default=0)
    parser.add_argument('-r', '--resume', type=bool, help="Resumes training from the last saved checkpoint.", default=False)
    parser.add_argument('-b', '--batch', type=int, default=128)
    parser.add_argument('-s', '--save', type=bool, default=True)
    parser.add_argument('-t', '--patience', type=int, default=100)
    parser.add_argument('-w', '--workers', type=int, default=8)
    parser.add_argument('-n', '--name', type=str, default=datetime.now().strftime("%Y%m%d%H%M%S"))
    parser.add_argument('-o', '--optimizer', type=str, default='auto')
    parser.add_argument('-g', '--plots', type=bool, default=True)
    parser.add_argument('-f', '--train_fire_txt', type=str, default='/home/gyeongju/wildfire/pytorch-lightning-smoke-detection/data/final_split/train_fires_final.txt')
    #parser.add_argument('-c', '--split_ratio', nargs='+', type=int, default=None)


    # Parse command-line arguments
    args = parser.parse_args()

    # store argument values in individual variables
    model = args.model
    mode = args.mode
    data_path = args.data_path
    epochs = args.epochs
    imgsz = args.imgsz
    device = args.device
    resume = args.resume
    batch = args.batch
    save = args.save
    patience = args.patience
    workers = args.workers
    name = args.name
    optimizer = args.optimizer
    plots = args.plots
    train_fire_txt = args.train_fire_txt

    SRC_PATH = os.path.dirname(os.path.abspath(__file__)) # path of folder # /home/gkim283/wildfire_detect/yolov8_src

    if mode == "detect": # for detection mode
        data_folder_list = os.listdir(data_path)
        if data_path.split("/")[-1] == "":
            data_name = data_path.split("/")[-2]
        else: 
            data_name = data_path.split("/")[-1]
        yaml_name = f"{data_name}.yaml"
        print(yaml_name)
        if "train" not in data_folder_list:
            print("DATASET is not SPLITED")
        if yaml_name not in data_folder_list:
            yaml_path = build_yaml(data_path)
        else:
            yaml_path = f'{data_path}/{yaml_name}'
    
    # run
    if mode == 'detect':
        detect_train_yolo(model, yaml_path, epochs, batch, imgsz, device, resume, save, patience, workers, name, optimizer, plots)
    elif mode == 'cls':
        cls_train_yolo(model, data_path, epochs, batch, imgsz, device, resume, save, patience, workers, name, optimizer, plots)
    else:
        print("unvalid mode")


    #save_configuration(SRC_PATH, model, mode, data_path, epochs, imgsz, device, resume, batch, save, patience, workers, name, optimizer, plots, train_fire_txt)
    print("saved configuration successfully")