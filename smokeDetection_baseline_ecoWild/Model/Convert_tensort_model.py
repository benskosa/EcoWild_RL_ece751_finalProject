import onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Load the modified ONNX model
onnx_model_path = "./yolov8l-cls_whole_224_best_new.onnx"
onnx_model = onnx.load(onnx_model_path)

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(TRT_LOGGER)
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(network_flags)
parser = trt.OnnxParser(network, TRT_LOGGER)

success = parser.parse(onnx_model.SerializeToString())
if not success:
    for error in range(parser.num_errors):
        print(parser.get_error(error))

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

# Define the input shape as fixed [1, 3, 224, 224] for batch size 1
input_name = network.get_input(0).name
network.get_input(0).shape = [1, 3, 224, 224]  # Batch size 1, fixed input size

# Build the serialized engine
try:
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print("Failed to create the TensorRT engine!")
    else:
        print("Successfully created the TensorRT engine!")
except Exception as e:
    print(f"Error during engine creation: {e}")

# Save the engine to a file
engine_file_path = "./yolov8l-cls_whole_224_best_new.trt"
with open(engine_file_path, "wb") as f:
    f.write(engine)
