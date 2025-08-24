# Quantization Pipeline

A modular quantization pipeline for converting and optimizing language models with TensorRT and serving them with OpenAI-compatible APIs.

## Project Structure

```
quant/
├── quant.py                    # Main entry point
├── configs/
│   └── llama38_trt_mxfp4.yml  # Configuration file
├── dataset/
│   └── speed_benchmark.json    # Benchmark dataset
└── quant/                      # Core package
    ├── __init__.py
    ├── client/                 # Client implementations
    │   ├── __init__.py
    │   └── speed_benchmark.py
    ├── convertor/              # Model converters
    │   ├── __init__.py
    │   └── onnx.py
    ├── interface/              # Serving interfaces
    │   ├── __init__.py
    │   └── server_openai.py
    └── trt/                    # TensorRT quantizers
        ├── __init__.py
        └── mxfp4.py
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install TensorRT (version depends on your CUDA version):
```bash
# For CUDA 11.x
pip install tensorrt-cu11

# For CUDA 12.x  
pip install tensorrt-cu12
```

## Usage

### Basic Usage

Run the complete pipeline with a configuration file:

```bash
python quant.py --config configs/llama38_trt_mxfp4.yml
```

### Configuration

The pipeline is configured using YAML files. Here's the structure:

```yaml
interface:
  type: server_openai    # Interface type
  host: "0.0.0.0"       # Server host
  port: 8000            # Server port
  
model:
  model_name: meta-llama/Meta-Llama-3-8B-Instruct
  cache_dir: "./model_cache"
  max_length: 2048
  conversion:
    type: onnx           # Converter type
    output_path: "./models/llama3.onnx"
    quantization:
      type: trt_mxfp4    # Quantizer type
      precision: fp4     # Precision mode
      output_path: "./models/llama3.trt"

client:
  type: speed_benchmark  # Client type
  dataset: "dataset/speed_benchmark.json"
  output: "results/llama38_mxfp4.json"
```

### Components

#### Converters
- `ONNXConverter`: Converts PyTorch models to ONNX format

#### Quantizers
- `MXFp4Quantizer`: TensorRT quantization with MXFP4/FP16 precision

#### Interfaces
- `ServerOpenAI`: OpenAI-compatible API server

#### Clients
- `SpeedBenchmarkClient`: Performance benchmarking client

## API Usage

Once the server is running, you can use it with OpenAI-compatible clients:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="llama3-trt",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    max_tokens=100
)

print(response.choices[0].message.content)
```

## Extending the Pipeline

### Adding New Converters

Create a new converter in `quant/convertor/`:

```python
class MyConverter:
    @classmethod
    def load_from_dict(cls, config: Dict[str, Any]) -> 'MyConverter':
        return cls(config)
    
    def convert(self) -> bool:
        # Implementation
        pass
```

### Adding New Quantizers

Create a new quantizer in `quant/trt/`:

```python
class MyQuantizer:
    @classmethod
    def load_from_dict(cls, config: Dict[str, Any]) -> 'MyQuantizer':
        return cls(config)
    
    def quantize(self, input_path: str) -> bool:
        # Implementation
        pass
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        # Implementation
        pass
```

### Adding New Interfaces

Create a new interface in `quant/interface/`:

```python
class MyInterface:
    @classmethod
    def load_from_dict(cls, config: Dict[str, Any]) -> 'MyInterface':
        return cls(config)
    
    def start(self):
        # Implementation
        pass
```

### Adding New Clients

Create a new client in `quant/client/`:

```python
class MyClient:
    @classmethod
    def load_from_dict(cls, config: Dict[str, Any]) -> 'MyClient':
        return cls(config)
    
    def run(self):
        # Implementation
        pass
```

## Performance Notes

- TensorRT optimization can take significant time (10+ minutes for large models)
- ONNX conversion requires sufficient GPU memory
- The pipeline includes automatic fallbacks (TensorRT → ONNX → PyTorch)
- Benchmark results are saved in JSON format for analysis

## Troubleshooting

1. **CUDA/TensorRT Issues**: Ensure compatible CUDA and TensorRT versions
2. **Memory Issues**: Reduce `max_batch_size` or `max_length` in config
3. **Model Access**: Ensure you have access to the Hugging Face model
4. **Port Conflicts**: Change the server port in the configuration

## License

This project is licensed under the MIT License.