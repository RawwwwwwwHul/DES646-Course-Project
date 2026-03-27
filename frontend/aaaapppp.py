import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
import time
import pickle
import mediapipe as mp

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

class ASLApp(tk.Tk):
    def __init__(self, model_path):
        super().__init__()
        self.title("GESTURA- Because every gesture deserves to be understoodr")
        self.geometry('800x400')
        
        self.recognizer = ASLTypingRecorder(model_path)
        self.camera_active = False
        self.frame = None
        self.predicted_letters = []
        self.capture_thread = None

        # Camera feed area
        self.label_frame = tk.LabelFrame(self, text="Camera Feed")
        self.label_frame.place(x=10, y=10, width=380, height=270)

        self.camera_label = tk.Label(self.label_frame, text="Camera inactive", width=45, height=15)
        self.camera_label.pack()

        self.start_button = tk.Button(self.label_frame, text="Start", command=self.toggle_camera)
        self.start_button.pack(pady=5)

        # Text display area
        self.text_frame = tk.LabelFrame(self, text="Text Output")
        self.text_frame.place(x=400, y=10, width=380, height=270)

        self.text_area = scrolledtext.ScrolledText(self.text_frame, wrap=tk.WORD, width=45, height=15, state=tk.DISABLED)
        self.text_area.pack()

        self.char_count_label = tk.Label(self.text_frame, text="0 characters")
        self.char_count_label.pack(anchor='e', padx=5)

        # Buttons for new session, save, view
        self.new_button = tk.Button(self, text="New", command=self.new_session)
        self.new_button.place(x=410, y=300, width=70)

        self.save_button = tk.Button(self, text="Save", command=self.save_text)
        self.save_button.place(x=490, y=300, width=70)

        self.view_button = tk.Button(self, text="View Saved", command=self.view_saved)
        self.view_button.place(x=570, y=300, width=90)

    def toggle_camera(self):
        if not self.camera_active:
            self.camera_active = True
            self.start_button.config(text='Stop')
            self.predicted_letters = []
            self.text_area.config(state=tk.NORMAL)
            self.text_area.delete('1.0', tk.END)
            self.text_area.config(state=tk.DISABLED)
            self.char_count_label.config(text="0 characters")
            self.capture_thread = threading.Thread(target=self.recognition_loop, daemon=True)
            self.capture_thread.start()
        else:
            self.camera_active = False
            self.start_button.config(text='Start')

    def recognition_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera")
            self.camera_active = False
            self.start_button.config(text='Start')
            return
        
        last_snap = 0
        display_width = 320
        display_height = 240
        while self.camera_active:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)  # Mirror image

            # Display frame in label
            frame_resized = cv2.resize(frame, (display_width, display_height))
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.config(image=imgtk)

            now = time.time()
            if now - last_snap >= 2:
                last_snap = now
                landmarks = self.recognizer.extract_landmarks(frame)
                if landmarks is not None:
                    letter = self.recognizer.recognize_letter(landmarks)
                    if letter:
                        self.predicted_letters.append(letter)
                        full_text = ''.join(self.predicted_letters)
                        self.text_area.config(state=tk.NORMAL)
                        self.text_area.delete('1.0', tk.END)
                        self.text_area.insert(tk.END, full_text)
                        self.text_area.config(state=tk.DISABLED)
                        self.char_count_label.config(text=f"{len(full_text)} characters")
            self.update_idletasks()
            time.sleep(0.03)  # Adjust to reduce CPU usage

        cap.release()
        self.camera_label.config(image='', text="Camera inactive")

    def new_session(self):
        if self.camera_active:
            self.toggle_camera()
        self.predicted_letters = []
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete('1.0', tk.END)
        self.text_area.config(state=tk.DISABLED)
        self.char_count_label.config(text="0 characters")

    def save_text(self):
        text = self.text_area.get('1.0', tk.END).strip()
        if not text:
            messagebox.showinfo("Info", "No text to save.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'w') as f:
                f.write(text)
            messagebox.showinfo("Saved", f"Text saved to {file_path}")

    def view_saved(self):
        file_path = filedialog.askopenfilename(
            defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'r') as f:
                content = f.read()
            if self.camera_active:
                self.toggle_camera()
            self.text_area.config(state=tk.NORMAL)
            self.text_area.delete('1.0', tk.END)
            self.text_area.insert(tk.END, content)
            self.text_area.config(state=tk.DISABLED)
            self.char_count_label.config(text=f"{len(content)} characters")

if __name__ == "__main__":
    model_path = "backend/modelyt1.p"  # change this path if needed
    app = ASLApp(model_path)
    app.mainloop()
