# Ball Detection and Tracking with OpenCV

This project uses **OpenCV** and **Python** to detect and track balls in video footage. It leverages both **Canny Edge Detection** and **Hough Circle Transform** to improve accuracy.

---

## 📽️ **Features**

- **Real-Time Detection**: Process live video streams or pre-recorded footage.
- **Noise Reduction**: Utilizes morphological operations for image filtering.
- **Multiple Detection Methods**: Combines Canny Edge Detection and Hough Circle Transform for robust ball detection.
- **Brightness Adaptation**: Adjusts thresholds based on image brightness levels.

---

## 🚀 **Quick Setup**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ball-detection-ai.git
   cd ball-detection-ai
   ```

2. **Install Dependencies**:
   ```bash
   pip install opencv-python numpy imutils matplotlib
   ```

3. **Run the Project**:
   ```bash
   python ball_detection.py --video path/to/your/video.mp4
   ```

---

## ⚙️ **Dependencies**

- `OpenCV` (for image processing)
- `NumPy` (for numerical operations)
- `Imutils` (for resizing)
- `Matplotlib` (for visualization)

---

## 🔽 **Code Overview**

- **Main Script**: `ball_detection.py`
  - Reads video input.
  - Converts frames to HSV and RGB color spaces.
  - Applies noise reduction using morphological operations.
  - Detects edges and circles using Canny and Hough Transform.
  - Highlights detected balls and displays results in real-time.

---

## 🖼️ **Demo**

![Ball Detection Demo](path/to/demo-image.png)
