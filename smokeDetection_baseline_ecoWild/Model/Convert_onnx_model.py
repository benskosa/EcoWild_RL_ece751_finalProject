import torch
import torch.onnx
import torchvision.models as models
from ultralytics import YOLO  # Import YOLO class from Ultralytics package

# If yolo
# Load the YOLOv8 model
model = YOLO('/home/elab02/wildfire/img/test_edge/yolov8l-cls_whole_224_last_new.pt')  # Replace with your YOLO model's path

# Export the model to ONNX
model.export(format="onnx")


#If resnet
num_classes = 2 #number of output classes
model = models.resnet34(pretrained=False)

model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load('./best_resnet34_model_epoch_3.pt', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

model.eval()

#Create a dummy input tensor with the appropriate size (3x224x224 for ResNet34)
dummy_input = torch.randn(1, 3, 224, 224)

#Export the model to ONNX format
torch.onnx.export(model, dummy_input, "best_resnet34_model_epoch_3.onnx", 
                  export_params=True,          # Store the trained parameters
                  opset_version=11,            # ONNX version to export to
                  do_constant_folding=True,    # Simplify the model by folding constants
                  input_names=['input'],       # Name of the input node
                  output_names=['output'],     # Name of the output node
                  dynamic_axes={'input': {0: 'batch_size'},  # Dynamic batch size
                                'output': {0: 'batch_size'}})

print("Model has been converted to ONNX and saved as 'best_resnet34_model_epoch_3.onnx'")
