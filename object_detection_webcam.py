"""
Real-time Object Detection using Webcam
Detects faces, eyes, and other objects in real-time
"""

import cv2
import numpy as np

class WebcamObjectDetector:
    def __init__(self):
        """Initialize webcam and load detection models"""
        self.cap = cv2.VideoCapture(0)
        
        # Load Haar Cascades
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
        
        # Check if cascades loaded
        if self.face_cascade.empty():
            raise Exception("Error loading face cascade")
    
    def detect_faces(self, frame):
        """Detect faces and facial features in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, 'Face', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # Extract face region
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                cv2.putText(roi_color, 'Eye', (ex, ey-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Detect smile
            smiles = self.smile_cascade.detectMultiScale(
                roi_gray, 
                scaleFactor=1.8, 
                minNeighbors=20
            )
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
                cv2.putText(roi_color, 'Smile', (sx, sy-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return frame, len(faces)
    
    def detect_motion(self, frame, prev_frame):
        """Detect motion between frames"""
        if prev_frame is None:
            return frame, []
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Apply threshold
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw motion areas
        motion_detected = []
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Filter small movements
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(frame, 'Motion', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                motion_detected.append((x, y, w, h))
        
        return frame, motion_detected
    
    def run(self, mode='face'):
        """
        Run detection in specified mode
        mode: 'face' for face detection, 'motion' for motion detection
        """
        print(f"Starting webcam object detection in {mode} mode...")
        print("Press 'q' to quit, 'f' for face mode, 'm' for motion mode")
        
        prev_frame = None
        current_mode = mode
        
        while True:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process based on mode
            if current_mode == 'face':
                processed_frame, count = self.detect_faces(frame)
                # Display count
                cv2.putText(processed_frame, f'Faces: {count}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            elif current_mode == 'motion':
                processed_frame, motion_areas = self.detect_motion(frame, prev_frame)
                # Display motion count
                cv2.putText(processed_frame, f'Motion Areas: {len(motion_areas)}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                prev_frame = frame.copy()
            
            # Display mode
            cv2.putText(processed_frame, f'Mode: {current_mode.upper()}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Webcam Object Detection', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                current_mode = 'face'
                print("Switched to face detection mode")
            elif key == ord('m'):
                current_mode = 'motion'
                print("Switched to motion detection mode")
                prev_frame = None
            elif key == ord('s'):
                # Save snapshot
                filename = f'snapshot_{current_mode}.jpg'
                cv2.imwrite(filename, processed_frame)
                print(f"Snapshot saved as {filename}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped")


def main():
    try:
        detector = WebcamObjectDetector()
        detector.run(mode='face')
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your webcam is connected")
        print("2. Check if another application is using the webcam")
        print("3. Try running with sudo if on Linux")


if __name__ == '__main__':
    main()
