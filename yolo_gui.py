import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import os

# ----------------------------
# Load YOLOv8 model (use pretrained or your custom best.pt)
# ----------------------------
model_path = "yolov8n.pt"  # change to "best.pt" if you have your own model
model = YOLO(model_path)

# ----------------------------
# GUI Setup
# ----------------------------
root = tk.Tk()
root.title("Drone Detection and safe zone")
root.geometry("900x700")
root.config(bg="#202020")

# Title label
tk.Label(root, text="🔍 Drone Detection and safe zone", font=("Arial", 20, "bold"), bg="#202020", fg="white").pack(pady=20)

# Canvas for image
canvas = tk.Label(root, bg="#2b2b2b")
canvas.pack(pady=10)

# ----------------------------
# Detection Function
# ----------------------------
def detect_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return
    
    # Run YOLO prediction
    results = model.predict(source=file_path, show=False)[0]
    
    # Read and draw detections on the image
    frame = cv2.imread(file_path)
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Save the detected image
    output_path = "detected_result.jpg"
    cv2.imwrite(output_path, frame)

    # Display result on Tkinter canvas
    img = Image.open(output_path)
    img = img.resize((640, 480))
    img_tk = ImageTk.PhotoImage(img)
    canvas.configure(image=img_tk)
    canvas.image = img_tk

    messagebox.showinfo("Detection Complete", f"✅ Detection done! Saved as {output_path}")

# ----------------------------
# Button
# ----------------------------
btn = tk.Button(root, text="Upload & Detect Image", command=detect_image,
                font=("Arial", 14), bg="#4CAF50", fg="white", relief="flat", padx=20, pady=10)
btn.pack(pady=20)

root.mainloop()
