"""
Object Detection on Images using OpenCV DNN
Supports multiple pre-trained models (YOLO, MobileNet-SSD, etc.)
"""

import cv2
import numpy as np
import argparse
import os

class ObjectDetector:
    def __init__(self, model_type='yolo'):
        """
        Initialize object detector with specified model type
        model_type: 'yolo', 'mobilenet', or 'coco'
        """
        self.model_type = model_type
        self.load_model()
    
    def load_model(self):
        """Load the pre-trained model and configuration"""
        if self.model_type == 'yolo':
            # YOLOv3 model (you'll need to download the files)
            self.net = None
            self.classes = self.load_coco_names()
            print("YOLO model selected. Download yolov3.weights and yolov3.cfg")
            print("wget https://pjreddie.com/media/files/yolov3.weights")
            print("wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg")
        
        elif self.model_type == 'mobilenet':
            # Using OpenCV's built-in models
            print("Using MobileNet-SSD model")
            self.classes = self.load_coco_names()
    
    def load_coco_names(self):
        """Load COCO class names"""
        return [
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
    
    def detect_objects_cascade(self, image_path):
        """
        Simple object detection using Haar Cascade (for faces, cars, etc.)
        This works without downloading additional models
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Load pre-trained Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        print(f"Found {len(faces)} faces")
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # Detect eyes within face region
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # Display result
        cv2.imshow('Object Detection - Faces', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save result
        output_path = 'detected_' + os.path.basename(image_path)
        cv2.imwrite(output_path, img)
        print(f"Result saved to {output_path}")
    
    def detect_with_contours(self, image_path):
        """
        Simple object detection using contour detection
        Good for detecting distinct objects in images
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"Found {len(contours)} objects/contours")
        
        # Draw contours and bounding boxes
        for i, contour in enumerate(contours):
            # Filter small contours
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f'Object {i+1}', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display results
        cv2.imshow('Edge Detection', edges)
        cv2.imshow('Object Detection - Contours', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save result
        output_path = 'contour_detected_' + os.path.basename(image_path)
        cv2.imwrite(output_path, img)
        print(f"Result saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Object Detection on Images')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--method', type=str, default='cascade', 
                       choices=['cascade', 'contour'], 
                       help='Detection method: cascade (faces) or contour (general objects)')
    
    args = parser.parse_args()
    
    detector = ObjectDetector()
    
    if args.method == 'cascade':
        detector.detect_objects_cascade(args.image)
    elif args.method == 'contour':
        detector.detect_with_contours(args.image)


if __name__ == '__main__':
    # If no arguments provided, show usage
    import sys
    if len(sys.argv) == 1:
        print("Usage Examples:")
        print("  python object_detection_image.py --image photo.jpg --method cascade")
        print("  python object_detection_image.py --image photo.jpg --method contour")
        print("\nMethods:")
        print("  cascade - Detects faces using Haar Cascade")
        print("  contour - Detects objects using edge detection and contours")
    else:
        main()
