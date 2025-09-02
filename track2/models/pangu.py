import onnx
import onnxruntime as ort


def define_pangu_onnx(model_type=6):
    model = onnx.load(f'/data/pangu/checkpoints/pangu_weather_{model_type}.onnx')

    # Set the behavier of onnxruntime
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena=False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    # Increase the number for faster inference and more memory consumption
    options.intra_op_num_threads = 1

    # Set the behavier of cuda provider
    cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

    # Initialize onnxruntime session for Pangu-Weather Models
    ort_session = ort.InferenceSession(f'/data/pangu/checkpoints/pangu_weather_{model_type}.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])

    return ort_session