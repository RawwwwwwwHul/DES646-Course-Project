import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress TensorFlow INFO and WARNING messages

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


class ASLTypingRecorder:
    def __init__(self, model_path='backend/modelyt1.p'):
        # Load trained model
        print("Loading trained model...")
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)
            self.model = model_dict['modelyt1']
        print(" Model loaded")
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        
        self.predicted_letters = []
        #self.prediction_history=[]
        #self.history_size=5
        
    def extract_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = []
            hand_landmarks = results.multi_hand_landmarks[0]
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return np.array(landmarks)
        return None
    
    def recognize_letter(self, landmarks):
        landmarks_reshaped = landmarks.reshape(1, -1)
        pred = self.model.predict(landmarks_reshaped)[0]
        return pred
    
    def run(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        print("\nStart showing your ASL letter to the camera.")
        print("A snapshot will be taken every 2 seconds and recorded.")
        print("Press 'n' to stop and display recorded letters.")
        print("Press 'q' to quit immediately without saving.\n")
        
        last_snapshot_time = 0
        running = True
        
        while running:
            ret, frame = cap.read()
            if not ret:
                print("Error reading from camera.")
                break
            
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "Press 'n' to stop capturing, 'q' to quit", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('ASL Typing Recorder', frame)
            
            current_time = time.time()
            # Take a snapshot every 3 seconds
            if current_time - last_snapshot_time >= 2:
                last_snapshot_time = current_time
                
                landmarks = self.extract_landmarks(frame)
                if landmarks is not None:
                    letter = self.recognize_letter(landmarks)
                    if letter=='space':
                        self.predicted_letters.append(" ")
                        print("Recorded: [space]")
                    elif letter == "del":
                        if self.predicted_letters:
                            self.predicted_letters.pop()
                            print("Recorded: [del]")
                    else:
                        self.predicted_letters.append(letter) 
                        print(f"Recorded={letter}")            
                else:
                    print("No hand detected - snapshot skipped.")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):
                print("\nStopping capture and displaying recorded letters...")
                running = False
            elif key == ord('q'):
                print("\nQuit without saving.")
                #self.predicted_letters = []
                running = False
        
        cap.release()
        cv2.destroyAllWindows()
        self.show_recorded_letters()
    
    def show_recorded_letters(self):
        if self.predicted_letters:
            typed_text = ''.join(self.predicted_letters)
            print("\nYour recorded ASL letters:")
            print(typed_text)
        else:
            print("No letters were recorded.")

def main():
    recorder = ASLTypingRecorder(model_path='backend/modelyt1.p')
    recorder.run(camera_id=0)

if __name__ == "__main__":
    main()
