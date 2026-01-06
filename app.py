"""
Web-based Object Detection using Flask
Access via browser at http://localhost:5000     
"""

from flask import Flask, render_template, Response
import cv2
import numpy as np
import urllib.request
import os
from threading import Lock

app = Flask(__name__)

class YOLOObjectDetector:
    def __init__(self, camera_index=0):
        """Initialize webcam and load YOLO model"""
        # Try different camera indices
        self.cap = None
        for idx in [camera_index, 2, 1, 0]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.cap = cap
                    print(f"✓ Camera {idx} initialized successfully")
                    break
                cap.release()
        
        if self.cap is None:
            raise Exception("No working camera found")
        
        # COCO class names - 80 objects that can be detected
        self.classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        # Single neon cyan color for everything
        self.neon_color = (255, 255, 0)  # Neon cyan (BGR)
        
        self.net = None
        self.model_loaded = False
        self.lock = Lock()
        
        # Try to load YOLO model
        self.load_yolo_model()
    
    def draw_corner_accents(self, img, x1, y1, x2, y2, color, length=20, thickness=3):
        """Draw sharp corner accents on bounding box"""
        # Top-left corner
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
        
        # Top-right corner
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
        
        # Bottom-left corner
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
        
        # Bottom-right corner
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)
    
    def draw_text_with_background(self, img, text, pos, font_scale=0.6, thickness=2, 
                                   text_color=(255, 255, 255), bg_color=(0, 0, 0), 
                                   padding=8, alpha=0.7):
        """Draw text with a semi-transparent sharp background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        x, y = pos
        # Create overlay for transparency
        overlay = img.copy()
        
        # Draw sharp rectangle background
        cv2.rectangle(overlay, 
                     (x - padding, y - text_height - padding),
                     (x + text_width + padding, y + baseline + padding),
                     bg_color, -1)
        
        # Blend overlay with original image
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Draw text
        cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        return text_height + baseline + padding * 2
    
    def download_file(self, url, filename):
        """Download file with progress"""
        if os.path.exists(filename):
            print(f"{filename} already exists, skipping download")
            return True
        
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded {filename} successfully!")
            return True
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            return False
    
    def load_yolo_model(self):
        """Load YOLOv4-tiny model (smaller and faster)"""
        # File paths
        weights_file = "yolov4-tiny.weights"
        config_file = "yolov4-tiny.cfg"
        
        # URLs for YOLOv4-tiny (much smaller than YOLOv3)
        weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
        config_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
        
        print("\n=== YOLO Model Setup ===")
        print("Checking for YOLO model files...")
        
        # Download files if needed
        weights_ok = self.download_file(weights_url, weights_file)
        config_ok = self.download_file(config_url, config_file)
        
        if weights_ok and config_ok:
            try:
                print("\nLoading YOLO model...")
                self.net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.model_loaded = True
                print("✓ YOLO model loaded successfully!")
                print(f"✓ Can detect {len(self.classes)} types of objects")
                print("======================\n")
            except Exception as e:
                print(f"Failed to load YOLO model: {e}")
                self.model_loaded = False
        else:
            print("\n⚠ Could not download YOLO model files")
            print("Falling back to simple object detection")
            print("======================\n")
    
    def detect_objects_yolo(self, frame):
        """Detect objects using YOLO and return with names"""
        if not self.model_loaded or self.net is None:
            return self.detect_objects_simple(frame)
        
        height, width = frame.shape[:2]
        
        # Prepare input blob
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Forward pass
        outputs = self.net.forward(output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:  # Confidence threshold
                    # Get bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        detected_objects = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                
                # Get class name
                class_id = class_ids[i]
                label = self.classes[class_id]
                confidence = confidences[i]
                color = self.neon_color
                
                # Draw sharp corner accents
                self.draw_corner_accents(frame, x, y, x + w, y + h, color, length=25, thickness=3)
                
                # Draw modern label with larger, nicer font
                label_text = f"{label.upper()} {int(confidence * 100)}%"
                
                # Position label above box with some margin
                label_y = max(y - 6, 20)
                self.draw_text_with_background(frame, label_text, (x, label_y), 
                                              font_scale=0.3, thickness=1,
                                              text_color=(0, 0, 0), 
                                              bg_color=color, padding=10, alpha=0.95)
                
                detected_objects.append((label, confidence, x, y, w, h))
        
        return frame, detected_objects
    
    def detect_objects_simple(self, frame):
        """Fallback simple object detection using contours"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 2000:
                x, y, w, h = cv2.boundingRect(contour)
                
                color = self.neon_color
                self.draw_corner_accents(frame, x, y, x+w, y+h, color, length=25, thickness=3)
                
                label = f'OBJECT {i+1}'
                self.draw_text_with_background(frame, label, (x, max(y - 15, 40)),
                                              font_scale=0.8, thickness=2,
                                              text_color=(0, 0, 0),
                                              bg_color=color, padding=10, alpha=0.95)
                
                detected_objects.append((f'Object {i+1}', 1.0, x, y, w, h))
        
        return frame, detected_objects
    
    def get_frame(self):
        """Get a processed frame for streaming"""
        with self.lock:
            ret, frame = self.cap.read()
            if not ret:
                return None
            
            # Detect objects
            processed_frame, objects = self.detect_objects_yolo(frame)
            
            # Add info overlay with nicer font
            neon = self.neon_color
            info_text = f'OBJECTS: {len(objects)}'
            cv2.putText(processed_frame, info_text, (15, 40), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.0, neon, 2)
            
            # Show detected objects list
            y_offset = 85
            if objects:
                cv2.putText(processed_frame, '[ DETECTED ]', (15, y_offset), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.7, neon, 2)
                y_offset += 35
                
                # Show unique object names
                unique_objects = {}
                for obj_name, conf, _, _, _, _ in objects:
                    if obj_name in unique_objects:
                        unique_objects[obj_name] += 1
                    else:
                        unique_objects[obj_name] = 1
                
                for obj_name, count in list(unique_objects.items())[:10]:  # Limit to 10
                    text = f"> {obj_name.upper()} x{count}" if count > 1 else f"> {obj_name.upper()}"
                    cv2.putText(processed_frame, text, (20, y_offset), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, neon, 2)
                    y_offset += 30
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            return buffer.tobytes()
    
    def __del__(self):
        if self.cap:
            self.cap.release()

# Initialize detector
detector = None

def init_detector():
    global detector
    if detector is None:
        try:
            detector = YOLOObjectDetector()
        except Exception as e:
            print(f"Error initializing detector: {e}")
            detector = None
    return detector

def generate_frames():
    """Generate frames for video streaming"""
    det = init_detector()
    if det is None:
        # Generate error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, 'Camera not available', (100, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return
    
    while True:
        frame = det.get_frame()
        if frame is None:
            break
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 NEON OBJECT DETECTION - WEB VERSION")
    print("="*60)
    print("\n📱 Open your browser and go to:")
    print("   👉 http://localhost:8080")
    print("   👉 http://127.0.0.1:8080")
    print("\n⚡ Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
