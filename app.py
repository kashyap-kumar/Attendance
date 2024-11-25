import tkinter as tk
from tkinter import messagebox, filedialog
import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime

# Directory to save data
DATA_DIR = "student_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Load existing data
def load_student_data():
    if os.path.exists(f"{DATA_DIR}/students.npy"):
        return np.load(f"{DATA_DIR}/students.npy", allow_pickle=True).item()
    return {}

students = load_student_data()

# Save data
def save_student_data():
    np.save(f"{DATA_DIR}/students.npy", students)

# Register a new student
def register_student():
    name = name_entry.get()
    roll_no = roll_no_entry.get()

    if not name or not roll_no:
        messagebox.showerror("Error", "Name and Roll No. are required!")
        return
    
    cam = cv2.VideoCapture(0)
    messagebox.showinfo("Info", "Look at the camera to register your face.")
    ret, frame = cam.read()
    cam.release()

    if ret:
        face_locations = face_recognition.face_locations(frame)
        if len(face_locations) == 1:
            face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
            students[roll_no] = {"name": name, "face_encoding": face_encoding}
            save_student_data()
            messagebox.showinfo("Success", "Student registered successfully!")
        else:
            messagebox.showerror("Error", "Ensure one face is visible.")
    else:
        messagebox.showerror("Error", "Failed to capture image.")

# Take attendance
def take_attendance():
    attendance = []
    cam = cv2.VideoCapture(0)
    messagebox.showinfo("Info", "Looking for registered faces...")
    for _ in range(10):  # Capture 10 frames
        ret, frame = cam.read()
        if not ret:
            continue

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                [student["face_encoding"] for student in students.values()],
                face_encoding
            )
            if True in matches:
                index = matches.index(True)
                roll_no = list(students.keys())[index]
                if roll_no not in attendance:
                    attendance.append(roll_no)
                    print(f"Marked present: {students[roll_no]['name']} ({roll_no})")

    cam.release()

    # Save attendance to file
    with open(f"{DATA_DIR}/attendance_{datetime.now().strftime('%Y%m%d')}.txt", "w") as file:
        for roll_no in attendance:
            file.write(f"{students[roll_no]['name']} ({roll_no})\n")
    messagebox.showinfo("Success", "Attendance captured.")

# Main UI
root = tk.Tk()
root.title("Attendance Management System")

tk.Label(root, text="Name:").grid(row=0, column=0, padx=10, pady=5)
name_entry = tk.Entry(root)
name_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Roll No:").grid(row=1, column=0, padx=10, pady=5)
roll_no_entry = tk.Entry(root)
roll_no_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Button(root, text="Register", command=register_student).grid(row=2, column=0, columnspan=2, pady=10)
tk.Button(root, text="Take Attendance", command=take_attendance).grid(row=3, column=0, columnspan=2, pady=10)

root.mainloop()
