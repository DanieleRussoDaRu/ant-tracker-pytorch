# 🐜 Ant Tracker PyTorch

A clean, modular **Computer Vision & Object Tracking** pipeline built with **PyTorch**, **YOLO**, and **OpenCV** to detect and track ants from video streams in real-time. Designed as a hands-on project demonstrating fundamental deep learning workflows, custom data ingestion, fine-tuning via transfer learning, and multi-object tracking.

---

## 📁 Repository Structure

```text
ant-tracker-pytorch/
│
├── data/                      # Local dataset storage (git-ignored)
│   ├── raw_videos/            # Original input video files (.mp4, .avi)
│   ├── images/                # Extracted frames split into train/val
│   │   ├── train/
│   │   └── val/
│   └── labels/                # Bounding box annotations in YOLO format (.txt)
│       ├── train/
│       └── val/
│
├── models/                    # Model checkpoints and weight storage
│   └── ant_detector_best.pt   # Fine-tuned PyTorch / YOLO model weights
│
├── src/                       # Modular Python source code
│   ├── __init__.py            # Package initialization
│   ├── data_loader.py         # Frame extraction & dataset formatting utilities
│   ├── train.py               # Fine-tuning loop using PyTorch & Ultralytics API
│   ├── tracker.py             # Custom Centroid / IoU Multi-Object Tracker
│   └── utils.py               # Bounding box drawing, trajectory plotting, and metrics
│
├── outputs/                   # Inference outputs and logs (git-ignored)
│   ├── processed_videos/      # Rendered video files with tracked IDs
│   └── trajectories.csv       # Extracted (x, y) coordinates & frame timestamps
│
├── main.py                    # Main execution script for video processing pipeline
├── dataset.yaml               # YOLO dataset configuration file (paths, class names)
├── requirements.txt           # Python dependencies and versions
└── README.md                  # Project documentation & usage guide
```

---

## 🛠️ Key Components & Workflow

1. **Data Preparation (`src/data_loader.py`)**  
   Extracts keyframes from raw video files to avoid dataset redundancy, formats images, and aligns annotations in standard YOLO format.

2. **Model Training & Fine-Tuning (`src/train.py`)**  
   Performs transfer learning starting from pre-trained YOLO weights (`yolov8n.pt`) fine-tuned specifically on ant detection tasks.

3. **Multi-Object Tracking (`src/tracker.py`)**  
   Associates detected bounding boxes across sequential video frames using IoU (Intersection over Union) / Euclidean centroid matching to assign persistent object IDs.

4. **Pipeline Execution (`main.py`)**  
   End-to-end command-line interface (CLI) to process raw input videos, overlay tracking visualizations, and export trajectory logs to CSV.

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/your-username/ant-tracker-pytorch.git
cd ant-tracker-pytorch
pip install -r requirements.txt
```

### 2. Run Tracking Pipeline
To process a video with a pre-trained checkpoint:
```bash
python main.py --input data/raw_videos/sample_ant.mp4 --weights models/ant_detector_best.pt
```

---

## 📄 License
Distributed under the **MIT License**. See `LICENSE` for details.
