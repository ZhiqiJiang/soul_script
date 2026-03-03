import numpy as np
import torch
import nvtx
from collections import OrderedDict
import tensorrt as trt

numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool


class TensorRTPredictor:
    """
    Implements inference for the EfficientDet TensorRT engine.
    """

    def __init__(self, **kwargs):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")
        engine_path = kwargs.get("model_path", None)
        self.debug = kwargs.get("debug", False)
        assert engine_path, f"model:{engine_path} must exist!"
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.tensors = OrderedDict()

        # TODO: 支持动态shape输入
        for idx in range(self.engine.num_io_tensors):
            name = self.engine[idx]
            is_input = self.engine.get_tensor_mode(name).name == "INPUT"
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            binding = {
                "index": idx,
                "name": name,
                "dtype": dtype,
                "shape": list(shape)
            }
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        self.allocate_max_buffers()

    def allocate_max_buffers(self, device="cuda"):
        nvtx.push_range("allocate_max_buffers", color="blue")
        # 目前仅支持 batch 维度的动态处理
        batch_size = 1
        feat_len = 1
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            shape = self.engine.get_tensor_shape(binding)
            is_input = self.engine.get_tensor_mode(binding).name == "INPUT"
            if -1 in shape:
                if is_input:
                    shape = self.engine.get_tensor_profile_shape(binding, 0)[-1]
                    batch_size = shape[0]
                    if len(shape) > 1:
                        feat_len = shape[1]
                else:
                    shape[0] = batch_size
                    if len(shape) > 1:
                        shape[1] = feat_len
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            tensor = torch.empty(
                tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]
            ).to(device=device)
            self.tensors[binding] = tensor
        nvtx.pop_range()

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        specs = []
        for i, o in enumerate(self.inputs):
            specs.append((o["name"], o['shape'], o['dtype']))
            if self.debug:
                print(f"trt input {i} -> {o['name']} -> {o['shape']}")
        return specs

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for i, o in enumerate(self.outputs):
            specs.append((o["name"], o['shape'], o['dtype']))
            if self.debug:
                print(f"trt output {i} -> {o['name']} -> {o['shape']}")
        return specs

    def adjust_buffer(self, feed_dict):
        nvtx.push_range("adjust_buffer", color="yellow")
        for name, buf in feed_dict.items():
            input_tensor = self.tensors[name]
            current_shape = list(buf.shape)
            # slices = tuple(slice(0, dim) for dim in current_shape)
            # input_tensor[slices].copy_(buf)
            input_tensor.view(-1)[:buf.numel()].copy_(buf.view(-1))
            self.context.set_input_shape(name, current_shape)
        nvtx.pop_range()

    def predict(self, feed_dict, stream):
        """
        Execute inference on a batch of images.
        :param data: A list of inputs as numpy arrays.
        :return A list of outputs as numpy arrays.
        """
        nvtx.push_range("set_tensors", color="green")
        self.adjust_buffer(feed_dict)
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        nvtx.pop_range()
        nvtx.push_range("execute", color="red")
        noerror = self.context.execute_async_v3(stream)
        if not noerror:
            raise ValueError("ERROR: inference failed.")
        torch.cuda.synchronize()
        nvtx.pop_range()
        return self.tensors

    def __del__(self):
        del self.engine
        del self.context
        del self.inputs
        del self.outputs
        del self.tensors


def get_predictor(**kwargs):
    predict_type = kwargs.get("predict_type", "trt")
    if predict_type == "trt":
        return TensorRTPredictor(**kwargs)