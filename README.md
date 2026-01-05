# Object Detection System

A comprehensive Python-based object detection system using OpenCV with multiple detection methods and modes.

## Features

- **Image Detection**: Detect objects in static images
- **Webcam Detection**: Real-time detection using your webcam
- **Video Detection**: Process video files and save results
- **Multiple Detection Methods**:
  - Face detection (Haar Cascade)
  - Eye detection
  - Smile detection
  - Motion detection
  - Contour-based object detection
  - Car detection

## Installation

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install opencv-python numpy
```

## Usage

### 1. Image Object Detection

Detect objects in images using different methods:

```bash
# Face detection
python object_detection_image.py --image photo.jpg --method cascade

# Contour-based detection (general objects)
python object_detection_image.py --image photo.jpg --method contour
```

**Output**: Creates a new image with detected objects highlighted

### 2. Webcam Object Detection

Real-time detection using your webcam:

```bash
python object_detection_webcam.py
```

**Controls**:
- `f` - Switch to face detection mode
- `m` - Switch to motion detection mode
- `s` - Save snapshot
- `q` - Quit

### 3. Video Object Detection

Process video files:

```bash
# Basic usage
python object_detection_video.py --video input.mp4

# Specify output file
python object_detection_video.py --video input.mp4 --output result.mp4

# Different detection types
python object_detection_video.py --video input.mp4 --type face
python object_detection_video.py --video input.mp4 --type car
python object_detection_video.py --video input.mp4 --type contour

# Process without displaying (faster)
python object_detection_video.py --video input.mp4 --no-display
```

## Detection Methods Explained

### 1. Haar Cascade (Face/Eye/Car Detection)
- Pre-trained classifiers that come with OpenCV
- Fast and efficient for specific objects
- Works well for faces, eyes, and cars
- No additional model downloads required

### 2. Contour Detection
- Uses edge detection to find object boundaries
- Good for detecting distinct objects
- Works with any type of object
- Best for high-contrast images

### 3. Motion Detection
- Detects movement between frames
- Compares consecutive frames
- Highlights areas with motion
- Useful for surveillance applications

## Examples

### Test with sample image:
```python
# Create a simple test
import cv2
import numpy as np

# Create a test image
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(img, 'Test Image', (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
cv2.imwrite('test_image.jpg', img)

# Run detection
# python object_detection_image.py --image test_image.jpg --method contour
```

### Test webcam (make sure your webcam is connected):
```bash
python object_detection_webcam.py
```

## Advanced: Using YOLO (requires additional setup)

For more advanced object detection with YOLO:

1. Download YOLO weights and config:
```bash
wget https://pjreddie.com/media/files/yolov3.weights
wget https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg
wget https://github.com/pjreddie/darknet/raw/master/data/coco.names
```

2. Modify the code to load YOLO model (code template included)

## Troubleshooting

### Webcam not working:
- Check if webcam is connected
- Try changing camera index in code: `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`
- On Linux, may need permissions: `sudo usermod -a -G video $USER`

### Video processing slow:
- Use `--no-display` flag to skip visualization
- Reduce video resolution before processing
- Use simpler detection methods (face vs contour)

### Import errors:
- Make sure virtual environment is activated
- Reinstall packages: `pip install --upgrade opencv-python numpy`

## Project Structure

```
python_pro/
├── venv/                          # Virtual environment
├── object_detection_image.py      # Image detection
├── object_detection_webcam.py     # Real-time webcam detection
├── object_detection_video.py      # Video file processing
└── README.md                      # This file
```

## Performance Tips

1. **For images**: Use cascade method for faces, contour for general objects
2. **For webcam**: Face detection is fastest, motion detection is lightweight
3. **For videos**: Use `--no-display` for faster processing
4. **For better accuracy**: Ensure good lighting and clear images

## Future Enhancements

- Deep learning models (YOLO, SSD, Faster R-CNN)
- Multiple object tracking
- Object counting and analytics
- Custom model training
- GPU acceleration

## License

Free to use and modify for educational and commercial purposes.
