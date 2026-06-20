# GESTURA

**Because every gesture deserves to be understood.**

GESTURA is a real-time ASL-to-text virtual keyboard. It uses landmark-based hand tracking and a lightweight classical ML model to translate American Sign Language alphabet signs into typed text — built as a course project (DES646) to explore practical, accessible applications of computer vision.

![Python](https://img.shields.io/badge/python-3.x-blue) ![scikit--learn](https://img.shields.io/badge/model-RandomForestClassifier-orange) ![status](https://img.shields.io/badge/status-proof--of--concept-yellow)

---

## Demo

Check out `Working demo.mp4` in this repo for a full walkthrough of GESTURA in action.

---

## Features

- **Real-time hand tracking** — MediaPipe extracts 21-point hand landmarks from a live webcam feed.
- **ASL alphabet recognition** — A RandomForestClassifier (scikit-learn) classifies 28 ASL signs with ~99% accuracy on held-out data.
- **Live confidence scoring** — On-screen feedback shows the predicted letter and the model's confidence in real time.
- **Virtual keyboard GUI** — A CustomTkinter interface turns recognized signs into typed text, with `space` and `delete` signs supported for full sentence construction.
- **Autocorrect** — Integrated spell-checking (via the `autocorrect` library's `Speller`) cleans up recognized letter sequences.
- **Save feature** — Typed output can be saved and revisited later.

---

## How It Works

1. **Capture** — The webcam feed is read frame-by-frame using OpenCV.
2. **Landmark extraction** — MediaPipe detects the hand in frame and extracts its landmark coordinates.
3. **Classification** — Landmarks are passed to a trained RandomForestClassifier, which predicts the corresponding ASL letter.
4. **Text output** — The predicted letter is appended to a running text buffer, displayed live in the GUI, and passed through autocorrect before being finalized.

---

## Datasets

Two datasets were used to train and compare models:

| Dataset | Source | Size used | Notes |
|---|---|---|---|
| ASL Alphabet (Kaggle) | Public dataset | ~1,400 images | Filtered via MediaPipe to keep only clear, visible hand signs |
| Custom webcam dataset | Self-recorded | ~610 images (~850/class collected) | Captured at a fixed 200x200 crop for consistent background, lighting, and hand orientation |

Both datasets were preprocessed by extracting hand landmarks with MediaPipe and serializing them (along with labels) into `.pickle` files for fast, consistent loading during training. The custom dataset model consistently outperformed the Kaggle-trained model in real-time, real-world testing.

---

## Tech Stack

| Category | Tools |
|---|---|
| Computer Vision | OpenCV, MediaPipe |
| ML | scikit-learn (RandomForestClassifier), NumPy, Pickle |
| GUI | CustomTkinter, Tkinter |
| Text Processing | autocorrect (Speller) |
| Misc | tqdm, threading, PIL |

> Note: Earlier iterations explored a deep learning approach (TensorFlow/Keras via Teachable Machine, with `cvzone` for hand detection), but this was dropped in favor of classical ML due to model export/compatibility issues and inconsistent accuracy across classes.

---

## Project Structure

```
DES646-Course-Project/
├── backend/      # Model training, landmark extraction, and inference logic
├── frontend/     # GUI application (entry point: Final.py)
├── dataset/      # Dataset(s) used for training
├── DES646_FINAL_REPORT.pdf   # Full project report
├── Working demo.mp4          # Demo video
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.x
- A webcam

### Setup

```bash
# Clone the repo
git clone https://github.com/RawwwwwwwHul/DES646-Course-Project.git
cd DES646-Course-Project

# Create and activate a virtual environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
python frontend/Final.py
```

> Make sure your virtual environment is activated and your webcam is accessible before running.

---

## Results

- **~99% classification accuracy** on both datasets (80/20 train-test split, evaluated with `accuracy_score`).
- The **custom dataset model outperformed the Kaggle-trained model** in live, real-world testing across all 28 signs.
- Known weak spots in real-time use:
  - **A** and **S** are sometimes misclassified as **E** due to visual similarity.
  - **O** is often confused with **C**.
  - **M**, **N**, and **T** occasionally go undetected if MediaPipe fails to register the hand.
  - **Q** shows inconsistent accuracy, likely due to camera quality or sign complexity.

---

## Limitations & Future Work

- Performance is sensitive to lighting, camera resolution, and signer variation.
- Currently limited to static, isolated ASL alphabet signs — no support for continuous sign sequences or dynamic gestures.
- The GUI is functional but minimal; further design refinement is planned.
- Future directions include: dataset augmentation for broader signer diversity, higher-resolution capture, and exploring transformer-based temporal models for continuous sign recognition.

---

## Acknowledgements

Built as a DES646 course project. Full methodology, research context, and reflections are documented in [`DES646_FINAL_REPORT.pdf`](./DES646_FINAL_REPORT.pdf).

