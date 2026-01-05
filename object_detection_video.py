"""
Object Detection on Video Files
Process video files and detect objects frame by frame
"""

import cv2
import numpy as np
import argparse
import os

class VideoObjectDetector:
    def __init__(self):
        """Initialize video detector with Haar Cascades"""
        # Load cascades
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.car_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_car.xml'
        )
        
        # Colors for different objects
        self.colors = {
            'face': (255, 0, 0),
            'eye': (0, 255, 0),
            'car': (0, 0, 255),
            'object': (255, 255, 0)
        }
    
    def detect_objects_in_frame(self, frame, detection_type='face'):
        """Detect objects in a single frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = []
        
        if detection_type == 'face':
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), self.colors['face'], 2)
                cv2.putText(frame, 'Face', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.colors['face'], 2)
                detections.append(('face', x, y, w, h))
                
                # Detect eyes within face
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), 
                                 self.colors['eye'], 2)
        
        elif detection_type == 'car':
            # Detect cars
            cars = self.car_cascade.detectMultiScale(gray, 1.1, 3)
            for (x, y, w, h) in cars:
                cv2.rectangle(frame, (x, y), (x+w, y+h), self.colors['car'], 2)
                cv2.putText(frame, 'Car', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.colors['car'], 2)
                detections.append(('car', x, y, w, h))
        
        elif detection_type == 'contour':
            # Detect objects using contours
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), self.colors['object'], 2)
                    detections.append(('object', x, y, w, h))
        
        return frame, detections
    
    def process_video(self, video_path, output_path=None, detection_type='face', show_video=True):
        """Process entire video file"""
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video Properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Detection Type: {detection_type}")
        
        # Setup video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = 0
        
        print("\nProcessing video... Press 'q' to stop")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            processed_frame, detections = self.detect_objects_in_frame(frame, detection_type)
            
            # Add frame info
            frame_count += 1
            total_detections += len(detections)
            cv2.putText(processed_frame, f'Frame: {frame_count}/{total_frames}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f'Detected: {len(detections)}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write to output video
            if out:
                out.write(processed_frame)
            
            # Display video
            if show_video:
                cv2.imshow('Video Object Detection', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Print progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per frame: {total_detections/frame_count:.2f}")
        
        if output_path:
            print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Object Detection on Video Files')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, help='Path to output video (optional)')
    parser.add_argument('--type', type=str, default='face', 
                       choices=['face', 'car', 'contour'],
                       help='Type of detection: face, car, or contour')
    parser.add_argument('--no-display', action='store_true', 
                       help='Process without displaying video')
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.video))[0]
        args.output = f'{base_name}_detected.mp4'
    
    detector = VideoObjectDetector()
    detector.process_video(
        args.video, 
        args.output, 
        args.type, 
        show_video=not args.no_display
    )


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        print("Usage Examples:")
        print("  python object_detection_video.py --video input.mp4")
        print("  python object_detection_video.py --video input.mp4 --output result.mp4")
        print("  python object_detection_video.py --video input.mp4 --type car")
        print("  python object_detection_video.py --video input.mp4 --no-display")
        print("\nDetection Types:")
        print("  face    - Detect faces and eyes")
        print("  car     - Detect cars")
        print("  contour - Detect objects using edge detection")
    else:
        main()
