import cv2
import os

# Folder to save captured images
save_dir = "dataset/Q"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

# Box parameters
BOX_SIZE = 200
print("Press 's' to save hand image, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    frame = cv2.flip(frame, 1)

    # Get frame dimensions
    h, w, _ = frame.shape

    # Define fixed 200x200 box at the center
    x1 = w // 2 - BOX_SIZE // 2
    y1 = h // 2 - BOX_SIZE // 2
    x2 = x1 + BOX_SIZE
    y2 = y1 + BOX_SIZE

    # Draw the fixed box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display instruction text
    cv2.putText(frame, "Place hand inside box", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Press 's' to save, 'q' to quit", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Show live feed
    cv2.imshow("Fixed Box Hand Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # Crop region inside the box
        cropped = frame[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (200, 200))

        filename = os.path.join(save_dir, f"hand_{count}.jpg")
        cv2.imwrite(filename, resized)
        print(f"Saved {filename}")
        count += 1

    elif key == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()



