"""
Real-time Object Detection using Webcam with YOLO
Detects and names specific objects like book, watch, bottle, phone, etc.
"""

import cv2
import numpy as np
import urllib.request
import os

class YOLOObjectDetector:
    def __init__(self):
        """Initialize webcam and load YOLO model"""
        self.cap = cv2.VideoCapture(0)
        
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
        
        # Generate random colors for each class
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype='uint8')
        
        self.net = None
        self.model_loaded = False
        
        # Try to load YOLO model
        self.load_yolo_model()
    
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
                
                # Get class name and color
                class_id = class_ids[i]
                label = self.classes[class_id]
                confidence = confidences[i]
                color = [int(c) for c in self.colors[class_id]]
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw label with confidence
                text = f"{label}: {confidence:.2f}"
                cv2.putText(frame, text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
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
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f'Object {i+1}', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                detected_objects.append((f'Object {i+1}', 1.0, x, y, w, h))
        
        return frame, detected_objects
    
    def run(self):
        """Run object detection"""
        print("\n=== Starting Object Detection ===")
        if self.model_loaded:
            print("Mode: YOLO - Detects and names specific objects")
            print("\nDetectable objects include:")
            print("person, book, laptop, cell phone, bottle, cup, chair, tv, clock,")
            print("keyboard, mouse, scissors, backpack, and 70+ more objects!")
        else:
            print("Mode: Simple contour detection")
        print("\nPress 'q' to quit")
        print("Press 's' to save snapshot")
        print("================================\n")
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            frame_count += 1
            
            # Detect objects (process every frame for real-time)
            processed_frame, objects = self.detect_objects_yolo(frame)
            
            # Display info
            info_text = f'Objects: {len(objects)}'
            cv2.putText(processed_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display FPS
            cv2.putText(processed_frame, f'Frame: {frame_count}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show detected objects list on the side
            y_offset = 90
            if objects:
                cv2.putText(processed_frame, 'Detected:', (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_offset += 25
                
                # Show unique object names
                unique_objects = {}
                for obj_name, conf, _, _, _, _ in objects:
                    if obj_name in unique_objects:
                        unique_objects[obj_name] += 1
                    else:
                        unique_objects[obj_name] = 1
                
                for obj_name, count in unique_objects.items():
                    text = f"{obj_name} x{count}" if count > 1 else obj_name
                    cv2.putText(processed_frame, text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 20
            
            cv2.imshow('Object Detection - Named Objects', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'detected_objects_{frame_count}.jpg'
                cv2.imwrite(filename, processed_frame)
                print(f"Snapshot saved as {filename}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nDetection stopped")


def main():
    try:
        detector = YOLOObjectDetector()
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Make sure your webcam is connected")
        print("2. Check if another application is using the webcam")
        print("3. Ensure you have internet connection for downloading model files")


if __name__ == '__main__':
    main()
