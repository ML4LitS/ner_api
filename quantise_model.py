from pathlib import Path
from optimum.onnxruntime import ORTModelForTokenClassification, ORTQuantizer
from transformers import AutoTokenizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

def quantize_and_save_model(original_model_path, quantized_model_path):
    # Load the original PyTorch model and convert it to ONNX format
    model = ORTModelForTokenClassification.from_pretrained(original_model_path, from_transformers=True)
    tokenizer = AutoTokenizer.from_pretrained(original_model_path)

    # Create the quantizer and setup the quantization configuration
    quantizer = ORTQuantizer.from_pretrained(model)
    dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

    # Quantize the model and save it to the specified path
    model_quantized_path = quantizer.quantize(
        save_dir=quantized_model_path,
        quantization_config=dqconfig,
    )
    print(f"Quantized model saved at: {model_quantized_path}")

# Example usage
original_model_path = '/home/stirunag/environments/models/run-cmry3ii0/best_model/'
quantized_model_path = Path(original_model_path).parent / "quantised"
quantize_and_save_model(original_model_path, quantized_model_path)
