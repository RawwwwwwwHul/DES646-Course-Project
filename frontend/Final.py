import customtkinter as ctk
import tkinter.filedialog as fd
from tkinter import messagebox
import threading
import cv2
import numpy as np
import time
from PIL import Image, ImageTk
import pickle
import mediapipe as mp
from autocorrect import Speller
from customtkinter import CTkImage

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


ctk.set_appearance_mode("light")  # 'dark' or 'light'
ctk.set_default_color_theme("blue")  # or 'green', 'dark-blue'

class ASLKeyboardApp(ctk.CTk):
    def __init__(self, model_path):
        super().__init__()
        self.title("GESTURA- Because every gesture deserves to be understood")
        self.geometry('950x450')
        self.resizable(False, False)

        self.spell=Speller('en')

        self.recognizer = ASLTypingRecorder(model_path)
        self.camera_active = False
        self.predicted_letters = []
        self.capture_thread = None
        self.display_width, self.display_height = 400, 300

        # Camera Card
        self.camera_card = ctk.CTkFrame(self, width=430, height=370)
        self.camera_card.place(x=20, y=40)
        self.camera_title = ctk.CTkLabel(self.camera_card, text="Camera Feed", font=ctk.CTkFont(size=18, weight='bold'))
        self.camera_title.pack(anchor="w", pady=(5,0), padx=10)
        self.camera_label = ctk.CTkLabel(self.camera_card, text="Camera inactive", width=self.display_width, height=self.display_height)
        self.camera_label.pack(pady=15)
        self.start_button = ctk.CTkButton(self.camera_card, text="Start", command=self.toggle_camera, width=240, height=40)
        self.start_button.pack(pady=5)

        # Output Card
        self.output_card = ctk.CTkFrame(self, width=430, height=370)
        self.output_card.place(x=500, y=40)
        self.output_title = ctk.CTkLabel(self.output_card, text="Text Output", font=ctk.CTkFont(size=18, weight='bold'))
        self.output_title.pack(anchor="w", pady=(5,0), padx=10)
        self.output_text = ctk.CTkTextbox(self.output_card, width=380, height=230, font=ctk.CTkFont(size=14))
        self.output_text.pack(pady=15)
        self.output_text.insert("1.0", "Recognized signs will appear here...")
        self.output_text.configure(state="disabled")
        self.char_count = ctk.CTkLabel(self.output_card, text="0 characters", anchor="e")
        self.char_count.pack(anchor="e", padx=10, pady=(0,2))

        # Controls
        self.new_button = ctk.CTkButton(self.output_card, text="New", width=95, command=self.new_session)
        self.new_button.place(x=0, y=298)
        self.save_button = ctk.CTkButton(self.output_card, text="Save", width=95, command=self.save_text)
        self.save_button.place(x=120, y=298)
        self.view_button = ctk.CTkButton(self.output_card, text="View Saved", width=120, command=self.view_saved)
        self.view_button.place(x=240, y=298)
    
    def autocorrect_text(self):
        original_text = self.output_text.get('1.0', 'end').strip()
        corrected_text = self.spell(original_text).upper()
        self.set_output(corrected_text)

    
    def toggle_camera(self):
        if not self.camera_active:
            self.camera_active = True
            self.start_button.configure(text="Stop")
            self.predicted_letters = []
            self.set_output("")
            self.capture_thread = threading.Thread(target=self.recognition_loop, daemon=True)
            self.capture_thread.start()
        else:
            self.camera_active = False
            self.start_button.configure(text="Start")
            self.autocorrect_text()

    def set_output(self, text):
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", 'end')
        self.output_text.insert('end', text)
        self.output_text.configure(state="disabled")
        self.char_count.configure(text=f"{len(text)} characters")

    def recognition_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera")
            self.camera_active = False
            return
        
        capture_interval=2.0
        last_snap = 0
        while self.camera_active:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            frame_resized = cv2.resize(frame, (self.display_width, self.display_height))

            now = time.time()
            time_left = capture_interval - (now - last_snap)
            if time_left < 0:
                time_left = 0

        # Draw timer countdown on the frame (white text, font scale 1, thickness 2)
            timer_text = f"{time_left:.1f}s"
            pos = (10, 30)  # Position near top-left corner
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame_resized, timer_text, pos, font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            
            img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            ctk_img = CTkImage(light_image=img, size=(self.display_width, self.display_height))
            self.camera_label.imgtk = ctk_img
            self.camera_label.configure(image=ctk_img, text="")

            now = time.time()
            if now - last_snap >= 2:
                last_snap = now
                landmarks = self.recognizer.extract_landmarks(frame)
                if landmarks is not None:
                    letter = self.recognizer.recognize_letter(landmarks)
                    if letter:
                        if letter=='space':
                            self.predicted_letters.append(' ')
                        elif letter=='del':
                            if self.predicted_letters:
                                self.predicted_letters.pop()
                        else:
                            self.predicted_letters.append(letter)
                        full_text = ''.join(self.predicted_letters)
                    self.set_output(full_text)
            self.update_idletasks()
            time.sleep(0.03)
        cap.release()
        self.camera_label.configure(image="", text="Camera inactive")

    def new_session(self):
        if self.camera_active:
            self.toggle_camera()
        self.predicted_letters = []
        self.set_output("")

    def save_text(self):
        text = self.output_text.get('1.0', 'end').strip()
        if not text:
            messagebox.showinfo("Info", "No text to save.")
            return
        file_path = fd.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'w') as f:
                f.write(text)
            messagebox.showinfo("Saved", f"Text saved to {file_path}")

    def view_saved(self):
        file_path = fd.askopenfilename(
            defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'r') as f:
                content = f.read()
            if self.camera_active:
                self.toggle_camera()
            self.set_output(content)

if __name__ == "__main__":
    model_path = "backend/modelyt1.p"
    app = ASLKeyboardApp(model_path)
    app.mainloop()
