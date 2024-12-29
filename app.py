import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime
import PIL.Image
import PIL.ImageTk
from PIL import Image
from threading import Thread
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from attendance_analytics import AttendanceAnalytics
from tkcalendar import DateEntry
import io

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
    batch = Column(String(50), nullable=False)  # e.g., "2023-25"
    course = Column(String(100), nullable=False)  # e.g., "MCA, B.Tech CSE"
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

        # Store paths of uploaded photos
        self.photo_paths = []

        self.create_gui()
        self.create_analytics_tab()
        self.create_attendance_list_tab()
        
    def create_gui(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Registration tab
        self.reg_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.reg_frame, text='Registration')
        
        # Registration form
        tk.Label(self.reg_frame, text="Student Registration", font=('Helvetica', 16, 'bold')).grid(row=0, column=0, columnspan=2, pady=10)
        
        # Create form fields
        form_fields = [
            ("Name:", "name_entry"),
            ("Roll No:", "roll_no_entry"),
            ("Batch:", "batch_entry"),
            ("Course:", "course_entry")
        ]
        
        for i, (label, attr) in enumerate(form_fields, start=1):
            tk.Label(self.reg_frame, text=label).grid(row=i, column=0, padx=5, pady=5, sticky='e')
            setattr(self, attr, tk.Entry(self.reg_frame))
            getattr(self, attr).grid(row=i, column=1, padx=5, pady=5, sticky='ew')
        
        # Add course dropdown menu with common courses
        self.course_var = tk.StringVar()
        common_courses = [
            "B.Tech CSE",
            "B.Tech IT",
            "B.Tech ECE",
            "B.Tech EEE",
            "B.Tech MECH",
            "B.Tech CIVIL",
            "M.Tech CSE",
            "MCA",
            "BCA"
        ]
        self.course_entry = ttk.Combobox(self.reg_frame, textvariable=self.course_var, values=common_courses)
        self.course_entry.grid(row=4, column=1, padx=5, pady=5, sticky='ew')
        
        # Add batch generation helper
        current_year = datetime.now().year
        batch_years = [f"{year}-{year+4}" for year in range(current_year-4, current_year+1)]
        self.batch_entry = ttk.Combobox(self.reg_frame, values=batch_years)
        self.batch_entry.set(f"{current_year}-{current_year+4}")  # Set default to current year
        
        # Preview frame
        self.preview_label = tk.Label(self.reg_frame)
        self.preview_label.grid(row=5, column=0, columnspan=2, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self.reg_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=10)
        
        tk.Button(button_frame, text="Start Preview", command=self.toggle_preview).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Register", command=self.register_student).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Clear Form", command=self.clear_registration_form).pack(side=tk.LEFT, padx=5)

        # Attendance tab
        self.att_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.att_frame, text='Attendance')
        
        # Create buttons frame
        buttons_frame = ttk.Frame(self.att_frame)
        buttons_frame.grid(row=0, column=0, columnspan=2, pady=10)
        
        tk.Button(buttons_frame, text="Take Live Attendance", command=self.take_attendance).pack(side=tk.LEFT, padx=5)
        tk.Button(buttons_frame, text="Upload Photos", command=self.upload_photos).pack(side=tk.LEFT, padx=5)
        tk.Button(buttons_frame, text="Process Uploaded Photos", command=self.process_uploaded_photos).pack(side=tk.LEFT, padx=5)
        tk.Button(buttons_frame, text="Clear Uploads", command=self.clear_uploads).pack(side=tk.LEFT, padx=5)
        tk.Button(buttons_frame, text="Export Report", command=self.export_attendance_report).pack(side=tk.LEFT, padx=5)
        
        # Create upload info label
        self.upload_info_label = tk.Label(self.att_frame, text="No photos uploaded")
        self.upload_info_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Attendance preview label (existing)
        self.att_preview_label = tk.Label(self.att_frame)
        self.att_preview_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        
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
            
        # Get all form values
        name = self.name_entry.get()
        roll_no = self.roll_no_entry.get()
        batch = self.batch_entry.get()
        course = self.course_entry.get()
        
        # Validate all fields
        if not all([name, roll_no, batch, course]):
            messagebox.showerror("Error", "All fields are required!")
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
                    batch=batch,
                    course=course,
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

    def clear_registration_form(self):
        """Clear all registration form fields"""
        self.name_entry.delete(0, tk.END)
        self.roll_no_entry.delete(0, tk.END)
        self.batch_entry.set('')  # Clear batch
        self.course_entry.set('')  # Clear course

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
        
        # Update tree columns to include new fields
        self.tree['columns'] = ('Roll No', 'Name', 'Batch', 'Course', 'Registration Date')
        
        # Configure all columns
        for col in self.tree['columns']:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
        
        session = SessionLocal()
        try:
            students = session.query(Student).all()
            for student in students:
                self.tree.insert('', 'end', values=(
                    student.roll_no,
                    student.name,
                    student.batch,
                    student.course,
                    student.registration_date
                ))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load student list: {str(e)}")
        finally:
            session.close()

    def create_analytics_tab(self):
        # Create Analytics Tab
        self.analytics_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analytics_frame, text='Analytics')
        
        # Initialize Analytics
        self.analytics = AttendanceAnalytics(engine)
        
        # Summary Statistics Frame
        summary_frame = ttk.LabelFrame(self.analytics_frame, text="Attendance Summary")
        summary_frame.pack(padx=10, pady=10, fill='x')
        
        # Get summary data
        summary = self.analytics.get_total_attendance_summary()
        
        # Display summary metrics
        metrics = [
            f"Total Students: {summary['total_students']}",
            f"Total Attendance Records: {summary['total_attendance_records']}",
            f"Unique Students Attended: {summary['unique_students_attended']}",
            f"Average Daily Attendance: {summary['average_daily_attendance']}",
            f"Attendance Percentage: {summary['attendance_percentage']}%"
        ]
        
        for i, metric in enumerate(metrics):
            ttk.Label(summary_frame, text=metric).grid(row=i//2, column=i%2, padx=5, pady=2, sticky='w')
        
        # Attendance Trend Plot
        trend_frame = ttk.LabelFrame(self.analytics_frame, text="Attendance Trend")
        trend_frame.pack(padx=10, pady=10, fill='both', expand=True)
        
        trend_df = self.analytics.generate_attendance_trend()
        trend_fig = self.analytics.plot_attendance_trend(trend_df)
        
        canvas = FigureCanvasTkAgg(trend_fig, master=trend_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Student Attendance Report
        report_frame = ttk.LabelFrame(self.analytics_frame, text="Student Attendance Report")
        report_frame.pack(padx=10, pady=10, fill='both', expand=True)
        
        # Treeview for student attendance report
        columns = ('Roll No', 'Name', 'Attendance Count', 'Attendance Percentage')
        student_report_tree = ttk.Treeview(report_frame, columns=columns, show='headings')
        
        for col in columns:
            student_report_tree.heading(col, text=col)
            student_report_tree.column(col, width=100)
        
        student_report_tree.pack(fill='both', expand=True)
        
        # Populate treeview
        student_df = self.analytics.generate_student_attendance_report()
        for index, row in student_df.iterrows():
            student_report_tree.insert('', 'end', values=list(row))
        
        # Export Button for Student Report
        export_button = ttk.Button(
            report_frame, 
            text="Export Student Report", 
            command=lambda: self.export_student_report(student_df)
        )
        export_button.pack(pady=5)

    # Add this method to the AttendanceSystem class
    def export_student_report(self, df):
        filename = filedialog.asksaveasfilename(
            defaultextension='.xlsx',
            filetypes=[("Excel files", "*.xlsx")]
        )
        if filename:
            df.to_excel(filename, index=False)
            messagebox.showinfo("Success", "Student attendance report exported successfully!")
    
    def create_attendance_list_tab(self):
        """
        Create a tab to display detailed attendance records
        """
        # Attendance List Tab
        self.attendance_list_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.attendance_list_frame, text='Attendance List')
        
        # Treeview for attendance records
        columns = ('Roll No', 'Name', 'Date', 'Time')
        self.attendance_tree = ttk.Treeview(self.attendance_list_frame, columns=columns, show='headings')
        
        # Configure column headings
        for col in columns:
            self.attendance_tree.heading(col, text=col)
            self.attendance_tree.column(col, width=150, anchor='center')
        
        # Pack the treeview
        self.attendance_tree.pack(fill='both', expand=True)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(self.attendance_list_frame, orient=tk.VERTICAL, command=self.attendance_tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.attendance_tree.configure(yscrollcommand=scrollbar.set)
        
        # Date filter frame
        filter_frame = ttk.Frame(self.attendance_list_frame)
        filter_frame.pack(fill='x', padx=10, pady=5)
        
        # Date range selection
        ttk.Label(filter_frame, text="From:").pack(side='left', padx=(0,5))
        self.from_date = DateEntry(filter_frame, width=12, background='darkblue', foreground='white', borderwidth=2, date_pattern='y-mm-dd')
        self.from_date.pack(side='left', padx=5)
        
        ttk.Label(filter_frame, text="To:").pack(side='left', padx=(10,5))
        self.to_date = DateEntry(filter_frame, width=12, background='darkblue', foreground='white', borderwidth=2, date_pattern='y-mm-dd')
        self.to_date.pack(side='left', padx=5)
        
        # Filter and Refresh buttons
        ttk.Button(filter_frame, text="Filter", command=self.filter_attendance).pack(side='left', padx=10)
        ttk.Button(filter_frame, text="Refresh", command=self.load_attendance_list).pack(side='left')
        
        # Initially load attendance list
        self.load_attendance_list()

    def load_attendance_list(self, start_date=None, end_date=None):
        """
        Load attendance records into the treeview
        
        Args:
            start_date (datetime, optional): Start date for filtering
            end_date (datetime, optional): End date for filtering
        """
        # Clear existing items
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)
        
        # Create a new session
        session = SessionLocal()
        
        try:
            # Base query to get attendance with student details
            query = (
                session.query(Attendance, Student)
                .join(Student, Attendance.roll_no == Student.roll_no)
            )
            
            # Apply date filtering if dates are provided
            if start_date and end_date:
                query = query.filter(
                    Attendance.date.between(start_date, end_date)
                )
            
            # Execute query and populate treeview
            for attendance, student in query.order_by(Attendance.date.desc()).all():
                self.attendance_tree.insert('', 'end', values=(
                    student.roll_no, 
                    student.name, 
                    attendance.date.strftime('%Y-%m-%d'), 
                    attendance.time.strftime('%H:%M:%S')
                ))
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load attendance list: {str(e)}")
        
        finally:
            session.close()

    def filter_attendance(self):
        """
        Filter attendance records based on date range
        """
        try:
            start_date = self.from_date.get_date()
            end_date = self.to_date.get_date()
            
            # Ensure end date is not before start date
            if end_date < start_date:
                messagebox.showerror("Error", "End date must be after start date")
                return
            
            # Load filtered attendance list
            self.load_attendance_list(start_date, end_date)
        
        except Exception as e:
            messagebox.showerror("Error", f"Error filtering attendance: {str(e)}")

    def clear_registration_form(self):
        self.name_entry.delete(0, tk.END)
        self.roll_no_entry.delete(0, tk.END)
    
    def upload_photos(self):
        """Allow users to select multiple photos for attendance"""
        filetypes = (
            ('Image files', '*.jpg *.jpeg *.png'),
            ('All files', '*.*')
        )
        
        filenames = filedialog.askopenfilenames(
            title='Select photos for attendance',
            filetypes=filetypes
        )
        
        if filenames:
            self.photo_paths.extend(filenames)
            self.update_upload_info()

    def clear_uploads(self):
        """Clear all uploaded photos"""
        self.photo_paths.clear()
        self.update_upload_info()
        self.att_preview_label.config(image='')

    def update_upload_info(self):
        """Update the upload information label"""
        if not self.photo_paths:
            self.upload_info_label.config(text="No photos uploaded")
        else:
            self.upload_info_label.config(text=f"{len(self.photo_paths)} photos uploaded")

    def process_uploaded_photos(self):
        """Process uploaded photos for attendance"""
        if not self.photo_paths:
            messagebox.showinfo("Info", "Please upload photos first!")
            return
        
        self.att_preview_label.config(text="Processing photos...")
        self.root.update()
        
        # Create a new thread for processing
        process_thread = Thread(target=self.process_photos_thread)
        process_thread.start()

    def process_photos_thread(self):
        """Thread function to process photos"""
        marked_students = set()
        session = SessionLocal()
        
        try:
            # Get all student data
            known_students = session.query(Student).all()
            known_encodings = [np.frombuffer(student.face_encoding, dtype=np.float64) for student in known_students]
            
            for photo_path in self.photo_paths:
                try:
                    # Load and convert image to numpy array
                    image = Image.open(photo_path)
                    image_array = np.array(image)
                    
                    # Update preview on main thread
                    self.root.after(0, self.update_photo_preview, image)
                    
                    # Detect faces and get encodings
                    face_locations = face_recognition.face_locations(image_array)
                    face_encodings = face_recognition.face_encodings(image_array, face_locations)
                    
                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces(known_encodings, face_encoding)
                        if True in matches:
                            index = matches.index(True)
                            student = known_students[index]
                            
                            if student.roll_no not in marked_students:
                                marked_students.add(student.roll_no)
                                self.mark_attendance(session, student.roll_no)
                
                except Exception as e:
                    print(f"Error processing photo {photo_path}: {str(e)}")
            
            # Show completion message on main thread
            self.root.after(0, self.show_photo_processing_complete, len(marked_students))
        
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        
        finally:
            session.close()

    def update_photo_preview(self, image):
        """Update preview with current photo being processed"""
        # Resize image for preview
        preview_size = (320, 240)
        image.thumbnail(preview_size, Image.Resampling.LANCZOS)
        photo = PIL.ImageTk.PhotoImage(image)
        self.att_preview_label.config(image=photo)
        self.att_preview_label.image = photo

    def show_photo_processing_complete(self, num_marked):
        """Show completion message after processing photos"""
        messagebox.showinfo("Success", f"Photo processing completed!\nMarked attendance for {num_marked} students.")
        self.att_preview_label.config(image='', text="Photo Processing Complete")
        self.photo_paths.clear()
        self.update_upload_info()

    def run(self):
        self.root.mainloop()

    def __del__(self):
        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    app = AttendanceSystem()
    app.run()