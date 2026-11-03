---
library_name: litert
pipeline_tag: image-classification
tags:
  - vision
  - image-classification
  - google
  - computer-vision
datasets:
  - imagenet-1k
model-index:
  - name: litert-community/resnet18
    results:
      - task:
          type: image-classification
          name: Image Classification
        dataset:
          name: ImageNet-1k
          type: imagenet-1k
          config: default
          split: validation
        metrics:
          - name: Top 1 Accuracy (Full Precision)
            type: accuracy
            value: 0.6977
          - name: Top 5 Accuracy (Full Precision)
            type: accuracy
            value: 0.8913
          - name: Top 1 Accuracy (Dynamic Quantized wi8 afp32)
            type: accuracy
            value: 0.6967
          - name: Top 5 Accuracy (Dynamic Quantized wi8 afp32)
            type: accuracy
            value: 0.8901
---

# ResNet 18

The ResNet-18 architecture is a convolutional neural network pre-trained on the ImageNet-1k dataset. Originally introduced by He et al. in the landmark paper, [**Deep Residual Learning for Image Recognition**](https://arxiv.org/pdf/1512.03385), this model utilizes residual mapping to overcome the vanishing gradient problem, enabling the training of substantially deeper networks.


## Model description

The model was converted from a checkpoint from PyTorch Vision. 

The original model has:    
acc@1 (on ImageNet-1K): 69.758%    
acc@5 (on ImageNet-1K): 89.078%    
num_params: 11,689,512   

## Available model files

| File | Description |
| --- | --- |
| `resnet18.tflite` | Full precision LiteRT/TFLite model. |
| `resnet18_dynamic_wi8_afp32.tflite` | Dynamic weight-only INT8 model with FP32 activations. |
| `resnet18_int8_channelwise.tflite` | Static INT8 model with channelwise INT8 weights and asymmetric INT8 activations. |

## Quantization

`resnet18_int8_channelwise.tflite` was produced with the `STATIC_WI8_AI8` quantization recipe. Weights are signed INT8 and use symmetric channelwise quantization for weight tensors. Activations are signed INT8 with asymmetric quantization parameters.

## Compatibility

| File | CPU | GPU | NPU |
|---|---|---|---|
| `resnet18.tflite` | Supported | Supported | N/A |
| `resnet18_int8_channelwise.tflite` | Supported | Not supported | Qualcomm / MediaTek |

## Intended uses & limitations

The model files were converted from pretrained weights from PyTorch Vision. The models may have their own licenses or terms and conditions derived from PyTorch Vision and the dataset used for training. It is your responsibility to determine whether you have permission to use the models for your use case.

## How to Use

**1. Install Dependencies** Ensure your Python environment is set up with the required libraries. Run the following command in your terminal:   

```bash 
pip install numpy Pillow huggingface_hub ai-edge-litert
```

**2. Prepare Your Image** The script expects an image file to analyze. Make sure you have an image (e.g., cat.jpg or car.png) saved in the same working directory as your script.

**3. Save the Script** Create a new file named `classify.py`, paste the script below into it, and save the file:

```python
#!/usr/bin/env python3
import argparse, json
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from ai_edge_litert.compiled_model import CompiledModel

def preprocess(img: Image.Image) -> np.ndarray:
   img = img.convert("RGB")
   w, h = img.size
   s = 256
   if w < h:
       img = img.resize((s, int(h * s / w)), Image.BILINEAR)
   else:
       img = img.resize((int(w * s / h), s), Image.BILINEAR)
   left = int(round((img.size[0] - 224) / 2.0))
   top = int(round((img.size[1] - 224) / 2.0))
   img = img.crop((left, top, left + 224, top + 224))

   x = np.asarray(img, dtype=np.float32) / 255.0
   x = (x - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
       [0.229, 0.224, 0.225], dtype=np.float32
   )
   return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

def main():
   ap = argparse.ArgumentParser()
   ap.add_argument("--image", required=True)
   args = ap.parse_args()

   model_path = hf_hub_download("litert-community/resnet18", "resnet18.tflite")
   labels_path = hf_hub_download(
       "huggingface/label-files", "imagenet-1k-id2label.json", repo_type="dataset"
   )
   with open(labels_path, "r", encoding="utf-8") as f:
       id2label = {int(k): v for k, v in json.load(f).items()}

   img = Image.open(args.image)
   x = preprocess(img)

   model = CompiledModel.from_file(model_path)
   inp = model.create_input_buffers(0)
   out = model.create_output_buffers(0)

   inp[0].write(x)
   model.run_by_index(0, inp, out)

   req = model.get_output_buffer_requirements(0, 0)
   y = out[0].read(req["buffer_size"] // np.dtype(np.float32).itemsize, np.float32)

   pred = int(np.argmax(y))
   label = id2label.get(pred, f"class_{pred}")

   print(f"Top-1 class index: {pred}")
   print(f"Top-1 label: {label}")
if __name__ == "__main__":
   main()
```

**4. Execute the Python Script**  Run the below command:

```bash 
python classify.py --image cat.jpg
```

### BibTeX entry and citation info

```bibtex
@inproceedings{he2016deep, 
title={Deep residual learning for image recognition}, 
author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian}, 
booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition}, pages={770--778}, 
year={2016} 
}
```
