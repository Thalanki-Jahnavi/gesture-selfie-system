# 📸 Gesture-Controlled Selfie Capture System

## 🚀 Overview

This project is a real-time computer vision application that captures selfies using hand gestures. It uses MediaPipe for hand tracking and OpenCV for image processing.

## ✨ Features

* ✋ Gesture-based selfie capture (Palm, Thumbs Up, Peace)
* ⏳ Countdown timer before capture
* 🎯 Face alignment detection for better framing
* 🧠 Reduced false positives (no smile/blink triggers)
* 📁 Saves images with gesture-based naming

## 🛠️ Tech Stack

* Python
* OpenCV
* MediaPipe
* NumPy

## ▶️ How to Run

```bash
pip install -r requirements.txt
python capture.py
```

Press **q** to exit.

## 📂 Output

Captured images are saved in the `selfies/` folder.

## 💡 Future Improvements

* Add voice feedback
* Add GUI interface
* Improve gesture accuracy

