
import os
import cv2
import mediapipe as mp
import pickle
import numpy as np
from tqdm import tqdm

class DatasetCreator:
    def __init__(self, dataset_dir='dataset'):
        """Initialize MediaPipe and dataset path"""
        self.dataset_dir = dataset_dir
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.3
        )
        
        print(f"Initialized dataset creator for: {dataset_dir}")
    
    def extract_landmarks(self, image_path):
        """Extract hand landmarks from an image"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.hands.process(image_rgb)
        
        # Extract landmarks if hand detected
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks)
        
        return None
    
    def create_dataset(self, output_file='new_data.pickle'):
        """Process all images and create pickle dataset"""
        data = []
        labels = []
        
        # Get all class folders (A, B, C, etc.)
        class_folders = sorted([f for f in os.listdir(self.dataset_dir) 
                               if os.path.isdir(os.path.join(self.dataset_dir, f))])
        
        print(f"\nFound {len(class_folders)} classes: {class_folders}")
        print("\nProcessing images...")
        
        total_processed = 0
        total_skipped = 0
        
        for class_name in class_folders:
            class_path = os.path.join(self.dataset_dir, class_name)
            
            # Get all images in this class folder
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            print(f"\nProcessing class '{class_name}': {len(image_files)} images")
            
            processed = 0
            skipped = 0
            
            # Process each image with progress bar
            for image_file in tqdm(image_files, desc=f"Class {class_name}"):
                image_path = os.path.join(class_path, image_file)
                
                # Extract landmarks
                landmarks = self.extract_landmarks(image_path)
                
                if landmarks is not None:
                    data.append(landmarks)
                    labels.append(class_name)
                    processed += 1
                else:
                    skipped += 1
            
            print(f"Processed: {processed},  Skipped (no hand detected): {skipped}")
            total_processed += processed
            total_skipped += skipped
        
        # Create dictionary
        data_dict = {
            'data': data,
            'labels': labels
        }
        
        # Save to pickle file
        with open(output_file, 'wb') as f:
            pickle.dump(data_dict, f)
        
        print(f"\n{'='*60}")
        print(f"Dataset creation complete!")
        print(f"{'='*60}")
        print(f"Total samples processed: {total_processed}")
        print(f"Total samples skipped: {total_skipped}")
        print(f"Success rate: {total_processed/(total_processed+total_skipped)*100:.1f}%")
        print(f"\nDataset saved as: {output_file}")
        print(f"Data shape: {len(data)} samples & {len(data[0])} features")
        print(f"Classes: {sorted(set(labels))}")
        
       
        print(f"\nClass distribution:")
        for class_name in sorted(set(labels)):
            count = labels.count(class_name)
            print(f"  {class_name}: {count} samples")
        
        return data_dict

def main():
    creator = DatasetCreator(dataset_dir='dataset')
    
    # Process images and create pickle file
    dataset = creator.create_dataset(output_file='new_data.pickle')
    
    print("\n Ready for training! Run: python train_asl_model.py")

if __name__ == "__main__":
    main()
