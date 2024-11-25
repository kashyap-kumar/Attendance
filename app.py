import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime
import PIL.Image
import PIL.ImageTk
from threading import Thread
import pandas as pd

# SQLAlchemy and MySQL dependencies
from sqlalchemy import create_engine, Column, String, DateTime, LargeBinary, Float, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session, relationship
from sqlalchemy.sql import func

# Database Configuration
DB_USERNAME = 'root'  # Replace with your MySQL username
DB_PASSWORD = 'password'  # Replace with your MySQL password
DB_HOST = 'localhost'
DB_NAME = 'attendance_system'

# Create SQLAlchemy base and engine
Base = declarative_base()
engine = create_engine(f'mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}', echo=False)
SessionLocal = scoped_session(sessionmaker(bind=engine))

# Database Models
class Student(Base):
    __tablename__ = 'students'
    
    roll_no = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    face_encoding = Column(LargeBinary, nullable=False)
    registration_date = Column(DateTime, server_default=func.now())
    
    # Relationship to Attendance
    attendances = relationship("Attendance", back_populates="student")

class Attendance(Base):
    __tablename__ = 'attendance'
    
    id = Column(String(50), primary_key=True)
    roll_no = Column(String(50), ForeignKey('students.roll_no'), nullable=False)
    date = Column(DateTime, nullable=False, server_default=func.now())
    time = Column(DateTime, nullable=False, server_default=func.now())
    
    # Relationship to Student
    student = relationship("Student", back_populates="attendances")

# Create tables
Base.metadata.create_all(engine)

class AttendanceSystem:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Advanced Attendance Management System")
        self.root.geometry("800x600")
        
        # Video capture variables
        self.cap = None
        self.preview_active = False
        self.current_frame = None
        
        self.create_gui()
        
    def create_gui(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Registration tab
        self.reg_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.reg_frame, text='Registration')
        
        # Registration form
        tk.Label(self.reg_frame, text="Student Registration", font=('Helvetica', 16, 'bold')).grid(row=0, column=0, columnspan=2, pady=10)
        
        tk.Label(self.reg_frame, text="Name:").grid(row=1, column=0, padx=5, pady=5)
        self.name_entry = tk.Entry(self.reg_frame)
        self.name_entry.grid(row=1, column=1, padx=5, pady=5)
        
        tk.Label(self.reg_frame, text="Roll No:").grid(row=2, column=0, padx=5, pady=5)
        self.roll_no_entry = tk.Entry(self.reg_frame)
        self.roll_no_entry.grid(row=2, column=1, padx=5, pady=5)
        
        # Preview frame
        self.preview_label = tk.Label(self.reg_frame)
        self.preview_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        
        # Buttons
        tk.Button(self.reg_frame, text="Start Preview", command=self.toggle_preview).grid(row=4, column=0, pady=10)
        tk.Button(self.reg_frame, text="Register", command=self.register_student).grid(row=4, column=1, pady=10)
        
        # Attendance tab
        self.att_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.att_frame, text='Attendance')
        
        # Attendance controls
        tk.Button(self.att_frame, text="Take Attendance", command=self.take_attendance).grid(row=0, column=0, pady=10)
        tk.Button(self.att_frame, text="Export Report", command=self.export_attendance_report).grid(row=0, column=1, pady=10)
        
        # Attendance preview
        self.att_preview_label = tk.Label(self.att_frame)
        self.att_preview_label.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        
        # Student list tab
        self.list_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.list_frame, text='Student List')
        
        # Treeview for student list
        self.tree = ttk.Treeview(self.list_frame, columns=('Roll No', 'Name', 'Registration Date'), show='headings')
        self.tree.heading('Roll No', text='Roll No')
        self.tree.heading('Name', text='Name')
        self.tree.heading('Registration Date', text='Registration Date')
        self.tree.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(self.list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Load student list
        self.load_student_list()

    def toggle_preview(self):
        if not self.preview_active:
            self.cap = cv2.VideoCapture(0)
            self.preview_active = True
            self.update_preview()
        else:
            self.preview_active = False
            if self.cap:
                self.cap.release()
            self.preview_label.config(image='')

    def update_preview(self):
        if self.preview_active:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (320, 240))
                photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                self.preview_label.config(image=photo)
                self.preview_label.image = photo
                self.root.after(10, self.update_preview)

    def register_student(self):
        if self.current_frame is None:
            messagebox.showerror("Error", "Please start preview first!")
            return
            
        name = self.name_entry.get()
        roll_no = self.roll_no_entry.get()
        
        if not name or not roll_no:
            messagebox.showerror("Error", "Name and Roll No. are required!")
            return
            
        face_locations = face_recognition.face_locations(self.current_frame)
        if len(face_locations) == 1:
            face_encoding = face_recognition.face_encodings(self.current_frame, face_locations)[0]
            
            # Save to database
            session = SessionLocal()
            try:
                new_student = Student(
                    roll_no=roll_no, 
                    name=name, 
                    face_encoding=face_encoding.tobytes()
                )
                session.add(new_student)
                session.commit()
                messagebox.showinfo("Success", "Student registered successfully!")
                self.load_student_list()
                self.clear_registration_form()
            except Exception as e:
                session.rollback()
                messagebox.showerror("Error", f"Registration failed: {str(e)}")
            finally:
                session.close()
        else:
            messagebox.showerror("Error", "Ensure one face is visible.")

    def take_attendance(self):
        self.cap = cv2.VideoCapture(0)
        self.att_preview_label.config(text="Taking Attendance...")
        
        # Create a new thread for attendance
        attendance_thread = Thread(target=self.process_attendance)
        attendance_thread.start()

    def process_attendance(self):
        marked_students = set()
        session = SessionLocal()
        
        try:
            # Get all student data
            known_students = session.query(Student).all()
            known_encodings = [np.frombuffer(student.face_encoding, dtype=np.float64) for student in known_students]
            
            while len(marked_students) < len(known_students):
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Update preview on main thread
                self.root.after(0, self.update_attendance_preview, frame)
                
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)
                
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                    if True in matches:
                        index = matches.index(True)
                        student = known_students[index]
                        
                        if student.roll_no not in marked_students:
                            marked_students.add(student.roll_no)
                            self.mark_attendance(session, student.roll_no)
                
                # Optional: break if all students marked
                if len(marked_students) == len(known_students):
                    break
            
            # Show completion message on main thread
            self.root.after(0, self.show_attendance_complete)
        
        except Exception as e:
            # Show error on main thread
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        
        finally:
            session.close()
            self.cap.release()

    def update_attendance_preview(self, frame):
        preview_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preview_frame = cv2.resize(preview_frame, (320, 240))
        photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(preview_frame))
        self.att_preview_label.config(image=photo)
        self.att_preview_label.image = photo

    def show_attendance_complete(self):
        messagebox.showinfo("Success", "Attendance completed!")
        self.att_preview_label.config(image='', text="Attendance Taken")

    def mark_attendance(self, session, roll_no):
        try:
            attendance_record = Attendance(
                id=f"{roll_no}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                roll_no=roll_no
            )
            session.add(attendance_record)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error marking attendance: {e}")

    def export_attendance_report(self):
        session = SessionLocal()
        
        try:
            # Get today's attendance with explicit join
            today = datetime.now().date()
            attendance_data = (
                session.query(Attendance)
                .join(Student, Attendance.roll_no == Student.roll_no)
                .filter(func.date(Attendance.date) == today)
                .all()
            )
            
            if not attendance_data:
                messagebox.showinfo("Info", "No attendance records found for today.")
                return
            
            # Prepare data for DataFrame
            report_data = [
                {
                    'Name': attendance.student.name, 
                    'Roll No': attendance.roll_no, 
                    'Date': attendance.date.date(), 
                    'Time': attendance.time.time()
                } 
                for attendance in attendance_data
            ]
            
            df = pd.DataFrame(report_data)
            
            # Save to Excel
            filename = filedialog.asksaveasfilename(
                defaultextension='.xlsx',
                filetypes=[("Excel files", "*.xlsx")]
            )
            if filename:
                df.to_excel(filename, index=False)
                messagebox.showinfo("Success", "Attendance report exported successfully!")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export report: {str(e)}")
        
        finally:
            session.close()

    def load_student_list(self):
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Create a new session
        session = SessionLocal()
        
        try:
            # Load students from database
            students = session.query(Student).all()
            for student in students:
                self.tree.insert('', 'end', values=(
                    student.roll_no, 
                    student.name, 
                    student.registration_date
                ))
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load student list: {str(e)}")
        
        finally:
            session.close()

    def clear_registration_form(self):
        self.name_entry.delete(0, tk.END)
        self.roll_no_entry.delete(0, tk.END)

    def run(self):
        self.root.mainloop()

    def __del__(self):
        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    app = AttendanceSystem()
    app.run()