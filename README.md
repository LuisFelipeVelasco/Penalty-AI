# ⚽ Penalty-AI

> **Using Computer Vision tools to solve a real sport problem — detecting whether a goalkeeper is off their line during a penalty kick.**

---

## 🎯 Purpose

In professional football, a penalty kick referee decision is one of the most disputed moments in the game. One of the key rules is that the **goalkeeper must remain on the goal line** until the ball is kicked. Detecting this violation in real time is difficult for human referees.

This project applies **Computer Vision** and **AI-powered object detection** to analyze penalty kick footage and automatically determine:

- 📍 Whether the **goalkeeper's feet are on or behind the goal line** at the moment of the kick
- ⚽ When the **ball is kicked** (significant position change detected)
- 🏁 The **exact position of the goal line** using image processing techniques

The core idea: combine classical Computer Vision (OpenCV) with modern deep learning (YOLO) to automate a sports officiating decision.

---

## 🛠️ Technologies & Libraries

| Library | Version | Role |
|---|---|---|
| `opencv-python` | ≥ 4.8.0 | Frame reading, color filtering, Canny edge detection, Hough Line Transform, ROI masking, drawing |
| `ultralytics` | ≥ 8.0.0 | YOLO models for object detection (ball) and pose estimation (goalkeeper keypoints) |
| `numpy` | ≥ 1.24.0 | Array operations, polygon masks, numerical computations |
| `sqlalchemy` | ≥ 2.0.0 | Utility import used in exploratory YOLO scripts |

### YOLO Models Used
- **`yolov8s.pt` / `yolov8m.pt`** — Object detection to locate the soccer ball (COCO class 32)
- **`yolo11m-pose.pt` / `yolo11l-pose.pt`** — Pose estimation to extract goalkeeper foot keypoints

---

## ⚙️ Setup

### Prerequisites
- Python **3.9 or higher**
- A machine with a webcam or video files in a `Videos/` folder
- (Optional but recommended) a GPU for faster YOLO inference

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Penalty-AI.git
cd Penalty-AI
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download YOLO model weights
The YOLO models are downloaded automatically by the `ultralytics` library on first run. Make sure you have an internet connection the first time you run the program. The models used are:
- `yolov8m.pt`
- `yolo11l-pose.pt`

### 5. Add your video files
Place your penalty kick video files inside a `Videos/` folder at the project root:
```
Penalty-AI/
└── Videos/
    ├── 14.mp4
    └── ...
```

### 6. Run the main program
```bash
python Program.py
```

The program will process the video frame by frame and pause automatically when it detects the **moment the ball is kicked**, displaying whether the goalkeeper was on the line or not.

---

## 🗂️ Project Structure

```
Penalty-AI/
│
├── Program.py                  # 🚀 Main entry point — full pipeline
│
├── open_cv_files/              # 📦 OpenCV exploration & building blocks
│   ├── read.py                 # Basic video capture and frame resizing
│   ├── basics_functions.py     # Grayscale, blur, Canny edge detection basics
│   ├── drawing.py              # Drawing text and shapes on frames
│   ├── detect_colors.py        # HSV color filtering to isolate white lines
│   ├── ROI.py                  # Region of Interest masking with polygons
│   ├── hough_transform.py      # Hough Line Transform experiments
│   └── lines_detection.py      # Refined goal line detection with slope filtering
│
├── Yolo_files/                 # 🤖 YOLO exploration & building blocks
│   ├── Yolo_basics.py          # Ball detection and kick moment detection
│   └── Yolo_pose.py            # Goalkeeper pose estimation & foot keypoints
│
├── Images/                     # Static test images
├── Videos/                     # Input penalty kick videos (not tracked by git)
├── requirements.txt            # Python dependencies
└── LICENSE
```

---

## 🧠 How It Works — Pipeline

```
Video Frame
    │
    ├──► [ROI: right half of frame]
    │        │
    │        └──► YOLO Pose (yolo11l-pose.pt)
    │                 │
    │                 └──► Extract left & right foot keypoints (x, y)
    │
    ├──► [ROI: goal area sub-region]
    │        │
    │        ├──► HSV Color Filter → isolate white pixels
    │        ├──► Dilation → thicken white regions
    │        ├──► Canny Edge Detection
    │        └──► Hough Line Transform → detect goal line
    │                 │
    │                 └──► Slope filter (1 < m < 2) + stability check
    │
    ├──► Compare foot Y-coordinates vs. goal line equation
    │        │
    │        └──► "Goalkeeper ON line" / "Goalkeeper NOT on line"
    │
    └──► YOLO Object Detection (yolov8m.pt) → track ball (class 32)
             │
             └──► Evaluate ball position change → detect kick moment → PAUSE
```

---

## 📚 Learnings

Working on this project provided hands-on experience with several important Computer Vision and AI concepts:

**Classical Computer Vision (OpenCV)**
- How to use **Region of Interest (ROI) masking** to focus processing on relevant parts of a frame, reducing noise and computation.
- **HSV color space** is far more robust than RGB for isolating specific colors (like white pitch lines) under varying lighting conditions.
- The **Hough Line Transform** can detect lines mathematically from edge images, but requires careful parameter tuning (threshold, `minLineLength`, `maxLineGap`) and post-processing (slope filtering) to be reliable.
- **Temporal stability filtering** — comparing new detections against previous frame values — is essential to prevent flickering and false positives caused by noise.
- The **Canny edge detector** combined with dilation produces clean, thick edges that feed well into the Hough transform.

**Deep Learning / YOLO**
- **YOLO pose estimation** exposes body keypoints (17 for the COCO body model), and the last two keypoints correspond to the feet — making it straightforward to extract precise foot positions.
- **YOLO object detection** with class filtering (class 32 = sports ball in COCO) allows tracking a specific object without a custom-trained model.
- Detecting **motion change** (ball kick moment) by comparing bounding box coordinates across frames is a lightweight and effective heuristic.
- Combining **two YOLO models** (detection + pose) in a single pipeline is computationally expensive; a GPU is strongly recommended.

**System Design**
- Separating the codebase into **exploratory scripts** (`open_cv_files/`, `Yolo_files/`) and a **final integrated program** (`Program.py`) is a good practice for iterative development in CV projects.
- Processing video **frame by frame** in a while loop gives full control over the analysis pipeline and allows pausing at key moments (the kick).

---

## 🚧 Known Limitations & Future Improvements

- The goal line detection is calibrated for a specific camera angle — a more robust solution would involve **camera calibration** and **homography** to normalize the perspective.

---

