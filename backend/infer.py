import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

class ASLDetector:
    def __init__(self, model_path='backend/modelyt1.p'):
        """Initialize the ASL detector with trained model"""
        
        # Load the trained model
        print("Loading trained model...")
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)
            self.model = model_dict['modelyt1']
        print(" Model loaded successfully")
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # For video stream
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # For FPS calculation
        self.prev_time = 0
        
        # For prediction smoothing
        self.prediction_history = []
        self.history_size = 5  # Number of frames to smooth over
        
    def extract_landmarks(self, image):
        """Extract hand landmarks from frame"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(image_rgb)
        
        # Extract landmarks if hand detected
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks), results.multi_hand_landmarks[0]
        
        return None, None
    
    def get_smoothed_prediction(self, prediction):
        """Smooth predictions over multiple frames to reduce jitter"""
        self.prediction_history.append(prediction)
        
        # Keep only recent predictions
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        # Return most common prediction
        if len(self.prediction_history) >= 3:
            return max(set(self.prediction_history), key=self.prediction_history.count)
        return prediction
    
    def run(self, camera_id=0):
        """Run real-time ASL detection"""
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "="*60)
        print("ASL Real-Time Detection Started")
        print("="*60)
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to toggle smoothing")
        print("  - Show your hand sign to the camera")
        print("="*60 + "\n")
        
        use_smoothing = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Extract landmarks
            landmarks, hand_landmarks = self.extract_landmarks(frame)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - self.prev_time) if self.prev_time else 0
            self.prev_time = current_time
            
            # Display info
            info_y = 30
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            info_y += 30
            smoothing_text = "ON" if use_smoothing else "OFF"
            cv2.putText(display_frame, f"Smoothing: {smoothing_text}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if landmarks is not None and hand_landmarks is not None:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Make prediction
                landmarks_reshaped = landmarks.reshape(1, -1)
                prediction = self.model.predict(landmarks_reshaped)[0]
                
                # Get prediction probabilities
                prediction_proba = self.model.predict_proba(landmarks_reshaped)[0]
                confidence = np.max(prediction_proba) * 100
                
                # Apply smoothing if enabled
                if use_smoothing:
                    prediction = self.get_smoothed_prediction(prediction)
                else:
                    self.prediction_history = []
                
                # Draw prediction box
                box_height = 100
                box_width = 400
                box_x = display_frame.shape[1] - box_width - 20
                box_y = 20
                
                # Semi-transparent background
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (box_x, box_y), 
                            (box_x + box_width, box_y + box_height),
                            (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
                
                # Draw border
                cv2.rectangle(display_frame, (box_x, box_y),
                            (box_x + box_width, box_y + box_height),
                            (0, 255, 0), 2)
                
                # Display prediction
                text_y = box_y + 40
                cv2.putText(display_frame, f"Sign: {prediction}",
                           (box_x + 20, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                text_y += 40
                cv2.putText(display_frame, f"Confidence: {confidence:.1f}%",
                           (box_x + 20, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
            else:
                # No hand detected
                cv2.putText(display_frame, "No hand detected",
                           (display_frame.shape[1] - 300, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Display instructions at bottom
            instructions_y = display_frame.shape[0] - 20
            cv2.putText(display_frame, "Press 'q' to quit | 's' to toggle smoothing",
                       (10, instructions_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('ASL Real-Time Detection', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                use_smoothing = not use_smoothing
                self.prediction_history = []  # Clear history when toggling
                print(f"Smoothing: {'ON' if use_smoothing else 'OFF'}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Detection stopped")

def main():
    """Main function to run ASL detection"""
    try:
        # Create detector instance
        detector = ASLDetector(model_path='backend/modelyt1.p')
        
        # Run detection
        detector.run(camera_id=0)  # Use 0 for default camera, 1 for external
        
    except FileNotFoundError:
        print("Error: model.p not found!")
        print("Please run train_asl_model.py first to train the model.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

