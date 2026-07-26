# Real-Time Object Detection with YOLO11 (Python and C++)

This project demonstrates **real-time object detection** using [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) and a webcam. Objects can be detected using either:
- The **pretrained YOLO11 model** (trained on the COCO dataset), or
- A **custom-trained YOLO11 model** (e.g., trained on the COCO8 dataset).

### Project Goals
- Provide side-by-side Python and C++ implementations of YOLO11 real-time detection.
- Show how to train YOLO11 on a custom dataset and run inference.
- Demonstrate ONNX portability across languages.

### Repository Structure
The repository contains **two implementations**:
- `python/` - Python code using Ultralytics YOLO and OpenCV (with a `custom_model/` folder for training).
- `cpp/` - C++ code using OpenCV’s DNN module with YOLO ONNX export.

#### Project Layout
```
real-time-object-detection-yolo/
├── python/
│   ├── main.py              # Real-time detection with pretrained/custom YOLO11
│   ├── requirements.txt     # Python dependencies
│   ├── custom_model/
│   │   └── custom_model.py  # Train YOLO11 on a custom dataset (e.g., COCO8)
│
├── cpp/
│   ├── main.cpp             # Real-time detection with YOLO ONNX + OpenCV
│   └── CMakeLists.txt       # Build instructions for C++
│
├── models/
│   └── yolo11n.pt           # YOLO11 weights (for Python)
│   └── yolo11n.onnx         # YOLO11 ONNX weights (for C++)
│
└── README.md                # Project overview

### Python Version

### Install dependencies
```bash
pip install -r python/requirements.txt

#### Run real-time detection:
python python/main.py

- Press q to quit.
- By default, it uses the pretrained YOLO11 model (yolo-Weights/yolo11n.pt).
- To use your own trained model, update the path in main.py.

#### Train a custom model
Inside python/custom_model/: python custom_model.py. This will train YOLO11 on a custom dataset (e.g., COCO8) 

### C++ Version
#### Build: 
cd cpp
mkdir build && cd build
cmake ..
make

#### Run: 
./yolo_cpp
- Uses the ONNX export of YOLO11 (models/yolo11n.onnx).
- Opens webcam and draws bounding boxes in real time.
- Press q to quit.

## Note:
- Python version uses Ultralytics YOLO11 directly.
- C++ version uses OpenCV DNN with the YOLO11 ONNX model.
- Both versions support pretrained COCO weights or custom-trained weights.
