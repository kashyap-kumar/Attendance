import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import cv2
import csv
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
from sqlalchemy import create_engine, Column, String, DateTime, LargeBinary, Float, ForeignKey, Enum
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session, relationship
from sqlalchemy.sql import func
import enum

# Database Configuration
DB_USERNAME = 'root'  # Replace with your MySQL username
DB_PASSWORD = 'password'  # Replace with your MySQL password
DB_HOST = 'localhost'
DB_NAME = 'attendance_system'

# Create SQLAlchemy base and engine
Base = declarative_base()
engine = create_engine(f'mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}', echo=False)
SessionLocal = scoped_session(sessionmaker(bind=engine))

# Add new Status enum
class AttendanceStatus(enum.Enum):
    PRESENT = "Present"
    ABSENT = "Absent"

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

class Subject(Base):
    __tablename__ = 'subjects'
    
    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    code = Column(String(20), nullable=False, unique=True)
    course = Column(String(100), nullable=False)
    batch = Column(String(50), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationship to Attendance
    attendances = relationship("Attendance", back_populates="subject")

class Attendance(Base):
    __tablename__ = 'attendance'
    
    id = Column(String(50), primary_key=True)
    roll_no = Column(String(50), ForeignKey('students.roll_no'), nullable=False)
    subject_id = Column(String(50), ForeignKey('subjects.id'), nullable=False)
    date = Column(DateTime, nullable=False, server_default=func.now())
    time = Column(DateTime, nullable=False, server_default=func.now())
    status = Column(Enum(AttendanceStatus), nullable=False, default=AttendanceStatus.PRESENT)
    
    # Relationships
    student = relationship("Student", back_populates="attendances")
    subject = relationship("Subject", back_populates="attendances")

class Marks(Base):
    __tablename__ = 'marks'
    
    id = Column(String(50), primary_key=True)
    roll_no = Column(String(50), ForeignKey('students.roll_no'), nullable=False)
    subject_code = Column(String(20), ForeignKey('subjects.code'), nullable=False)
    marks = Column(Float, nullable=False)
    test_code = Column(String(50), nullable=False)  # e.g., CT1, CT2
    
    # Relationships
    student = relationship("Student")
    subject = relationship("Subject")

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

        # Add common_courses and batch_years as instance variables
        current_year = datetime.now().year
        self.common_courses = [
            "B.Tech CSE", "B.Tech IT", "B.Tech ECE", "B.Tech EEE",
            "B.Tech MECH", "B.Tech CIVIL", "M.Tech CSE", "MCA", "BCA"
        ]
        self.batch_years = [f"{year}-{year+4}" for year in range(current_year-4, current_year+1)]
        
        # Add current_subject attribute
        self.current_subject = None
        
        self.create_gui()
        self.create_analytics_tab()
        self.create_attendance_list_tab()
        self.create_subject_tab()
        self.create_marks_tab()
        
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
        
        # Add buttons frame above the tree
        buttons_frame = ttk.Frame(self.list_frame)
        buttons_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        # Add Delete button
        delete_btn = ttk.Button(buttons_frame, text="Delete Selected", command=self.delete_selected_student)
        delete_btn.pack(side=tk.LEFT, padx=5)
        
        # Add Refresh button
        refresh_btn = ttk.Button(buttons_frame, text="Refresh List", command=self.load_student_list)
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # Treeview for student list
        self.tree = ttk.Treeview(self.list_frame, columns=('Roll No', 'Name', 'Batch', 'Course', 'Registration Date'), show='headings')
        self.tree.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        
        # Configure column headings and widths
        column_widths = {
            'Roll No': 100,
            'Name': 150,
            'Batch': 100,
            'Course': 150,
            'Registration Date': 150
        }
        
        for col, width in column_widths.items():
            self.tree.heading(col, text=col)
            self.tree.column(col, width=width)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(self.list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.grid(row=1, column=1, sticky='ns')
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Configure grid weights for proper resizing
        self.list_frame.grid_columnconfigure(0, weight=1)
        self.list_frame.grid_rowconfigure(1, weight=1)
        
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
        """Take attendance with subject selection"""
        # Create subject selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Subject")
        dialog.geometry("300x150")
        
        ttk.Label(dialog, text="Select Subject:").pack(pady=5)
        subject_var = tk.StringVar()
        
        # Get subjects from database
        session = SessionLocal()
        subjects = session.query(Subject).all()
        session.close()
        
        subject_combo = ttk.Combobox(dialog, 
                                    textvariable=subject_var,
                                    values=[f"{s.code} - {s.name}" for s in subjects])
        subject_combo.pack(pady=5)
        
        def start_attendance():
            if not subject_var.get():
                messagebox.showerror("Error", "Please select a subject!")
                return
            
            selected_subject = subjects[[f"{s.code} - {s.name}" for s in subjects].index(subject_var.get())]
            dialog.destroy()
            
            self.current_subject = selected_subject
            self.cap = cv2.VideoCapture(0)
            self.att_preview_label.config(text=f"Taking Attendance for {selected_subject.name}")
            
            # Create attendance thread
            attendance_thread = Thread(target=self.process_attendance)
            attendance_thread.start()
        
        ttk.Button(dialog, text="Start Attendance", command=start_attendance).pack(pady=10)

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
        """Mark attendance with subject reference"""
        try:
            attendance_record = Attendance(
                id=f"{roll_no}_{self.current_subject.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                roll_no=roll_no,
                subject_id=self.current_subject.id,
                status=AttendanceStatus.PRESENT
            )
            session.add(attendance_record)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error marking attendance: {e}")

    def export_attendance_report(self):
        session = SessionLocal()
        try:
            # Get subject selection from user
            dialog = tk.Toplevel(self.root)
            dialog.title("Select Subject")
            dialog.geometry("300x150")
            dialog.grab_set()  # Make the dialog modal
            
            ttk.Label(dialog, text="Select Subject:").pack(pady=5)
            subject_var = tk.StringVar()
            
            subjects = session.query(Subject).all()
            if not subjects:
                messagebox.showerror("Error", "No subjects available!")
                dialog.destroy()
                return
            
            subject_combo = ttk.Combobox(dialog, textvariable=subject_var, values=[f"{s.code} - {s.name}" for s in subjects])
            subject_combo.pack(pady=5)
            
            def generate_report():
                if not subject_var.get():
                    messagebox.showerror("Error", "Please select a subject!")
                    return
                
                selected_subject = subjects[[f"{s.code} - {s.name}" for s in subjects].index(subject_var.get())]
                dialog.destroy()
                self._generate_subject_report(selected_subject)
            
            ttk.Button(dialog, text="Generate Report", command=generate_report).pack(pady=10)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to select subject: {str(e)}")
        finally:
            session.close()

    def _generate_subject_report(self, subject):
        session = SessionLocal()
        try:
            # Get all students and their attendance for the selected subject
            students = (
                session.query(
                    Student.roll_no,
                    Student.name,
                    func.count(Attendance.id).label("present_count"),
                )
                .outerjoin(Attendance, (Student.roll_no == Attendance.roll_no) & (Attendance.subject_id == subject.id))
                .group_by(Student.roll_no)
                .all()
            )
            
            total_sessions = (
                session.query(func.count(Attendance.id))
                .filter(Attendance.subject_id == subject.id)
                .scalar()
            ) or 0  # Default to 0 if no sessions
            
            # Prepare data
            report_data = [
                {
                    "Roll No": student.roll_no,
                    "Name": student.name,
                    "Attendance Percentage": (student.present_count / total_sessions) * 100 if total_sessions > 0 else 0.0,
                    # "Status": "Present" if student.present_count > 0 else "Absent"
                }
                for student in students
            ]
            
            df = pd.DataFrame(report_data)
            
            # Save to Excel
            filename = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")],
            )
            if filename:
                df.to_excel(filename, index=False)
                messagebox.showinfo("Success", f"Attendance report for {subject.name} exported successfully!")
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
        """Create a tab to display detailed attendance records"""
        # Attendance List Tab
        self.attendance_list_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.attendance_list_frame, text='Attendance List')
        
        # Add filter frame
        filter_frame = ttk.Frame(self.attendance_list_frame)
        filter_frame.pack(fill='x', padx=10, pady=5)
        
        # Date range selection
        ttk.Label(filter_frame, text="From:").pack(side=tk.LEFT, padx=(0,5))
        self.from_date = DateEntry(filter_frame, width=12, background='darkblue', 
                                foreground='white', borderwidth=2, date_pattern='y-mm-dd')
        self.from_date.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(filter_frame, text="To:").pack(side=tk.LEFT, padx=(10,5))
        self.to_date = DateEntry(filter_frame, width=12, background='darkblue', 
                            foreground='white', borderwidth=2, date_pattern='y-mm-dd')
        self.to_date.pack(side=tk.LEFT, padx=5)
        
        # Subject filter
        ttk.Label(filter_frame, text="Subject:").pack(side=tk.LEFT, padx=(10,5))
        self.subject_filter = ttk.Combobox(filter_frame, width=20)
        self.subject_filter.pack(side=tk.LEFT, padx=5)
        self.update_subject_filter()
        
        # Filter and Refresh buttons
        ttk.Button(filter_frame, text="Filter", command=self.filter_attendance).pack(side=tk.LEFT, padx=10)
        ttk.Button(filter_frame, text="Refresh", command=self.load_attendance_list).pack(side=tk.LEFT)
        
        # Treeview for attendance records
        columns = ('Roll No', 'Name', 'Subject', 'Date', 'Time', 'Status')
        self.attendance_tree = ttk.Treeview(self.attendance_list_frame, columns=columns, show='headings')
        
        # Configure column headings
        for col in columns:
            self.attendance_tree.heading(col, text=col)
            self.attendance_tree.column(col, width=100)
        
        # Pack the treeview with scrollbar
        self.attendance_tree.pack(fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(self.attendance_list_frame, orient=tk.VERTICAL, 
                                command=self.attendance_tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.attendance_tree.configure(yscrollcommand=scrollbar.set)
        
        # Initially load attendance list
        self.load_attendance_list()

    def update_subject_filter(self):
        """Update the subject filter combobox with current subjects"""
        session = SessionLocal()
        try:
            subjects = session.query(Subject).all()
            self.subject_filter['values'] = ['All'] + [f"{s.code} - {s.name}" for s in subjects]
            self.subject_filter.set('All')
        finally:
            session.close()

    def load_attendance_list(self, start_date=None, end_date=None, subject_filter=None):
        """Load attendance records into the treeview"""
        # Clear existing items
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)
        
        session = SessionLocal()
        try:
            # Base query with joins
            query = (
                session.query(Attendance, Student, Subject)
                .join(Student, Attendance.roll_no == Student.roll_no)
                .join(Subject, Attendance.subject_id == Subject.id)
            )
            
            # Apply filters
            if start_date and end_date:
                query = query.filter(Attendance.date.between(start_date, end_date))
            
            if subject_filter and subject_filter != 'All':
                subject_code = subject_filter.split(' - ')[0]
                query = query.filter(Subject.code == subject_code)
            
            # Execute query and populate treeview
            for attendance, student, subject in query.order_by(Attendance.date.desc()).all():
                self.attendance_tree.insert('', 'end', values=(
                    student.roll_no,
                    student.name,
                    f"{subject.code} - {subject.name}",
                    attendance.date.strftime('%Y-%m-%d'),
                    attendance.time.strftime('%H:%M:%S'),
                    attendance.status.value
                ))
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load attendance list: {str(e)}")
        
        finally:
            session.close()

    def filter_attendance(self):
        """Filter attendance records based on date range and subject"""
        try:
            start_date = self.from_date.get_date()
            end_date = self.to_date.get_date()
            subject_filter = self.subject_filter.get()
            
            # Ensure end date is not before start date
            if end_date < start_date:
                messagebox.showerror("Error", "End date must be after start date")
                return
            
            # Load filtered attendance list
            self.load_attendance_list(start_date, end_date, subject_filter)
        
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
        """Process uploaded photos for attendance with subject selection"""
        if not self.photo_paths:
            messagebox.showinfo("Info", "Please upload photos first!")
            return
        
        # Create subject selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Subject")
        dialog.geometry("300x150")
        dialog.grab_set()  # Make dialog modal
        
        ttk.Label(dialog, text="Select Subject:").pack(pady=5)
        subject_var = tk.StringVar()
        
        # Get subjects from database
        session = SessionLocal()
        subjects = session.query(Subject).all()
        session.close()
        
        if not subjects:
            messagebox.showerror("Error", "No subjects found. Please add subjects first!")
            dialog.destroy()
            return
        
        subject_combo = ttk.Combobox(dialog, 
                                    textvariable=subject_var,
                                    values=[f"{s.code} - {s.name}" for s in subjects])
        subject_combo.pack(pady=5)
        
        def start_processing():
            if not subject_var.get():
                messagebox.showerror("Error", "Please select a subject!")
                return
            
            selected_subject = subjects[[f"{s.code} - {s.name}" for s in subjects].index(subject_var.get())]
            dialog.destroy()
            
            self.current_subject = selected_subject
            self.att_preview_label.config(text=f"Processing photos for {selected_subject.name}...")
            self.root.update()
            
            # Create a new thread for processing
            process_thread = Thread(target=self.process_photos_thread)
            process_thread.start()
        
        ttk.Button(dialog, text="Start Processing", command=start_processing).pack(pady=10)

    def process_photos_thread(self):
        """Thread function to process photos"""
        marked_students = set()
        session = SessionLocal()
        
        try:
            # Get all student data
            known_students = session.query(Student).all()
            known_encodings = [np.frombuffer(student.face_encoding, dtype=np.float64) for student in known_students]
            
            # Dictionary to track attendance status for each student
            attendance_status = {student.roll_no: False for student in known_students}
            
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
                                attendance_status[student.roll_no] = True
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
        messagebox.showinfo(
            "Success", 
            f"Photo processing completed!\n"
            f"Present: {num_marked} students\n"
            f"Absent: {self.get_total_students() - num_marked} students"
        )
        self.att_preview_label.config(image='', text=f"Photo Processing Complete - {self.current_subject.name}")
        self.photo_paths.clear()
        self.update_upload_info()

    def get_total_students(self):
        """Get total number of students in the database"""
        session = SessionLocal()
        try:
            return session.query(Student).count()
        finally:
            session.close()

    def delete_selected_student(self):
        """Delete the selected student from the database"""
        # Get selected item
        selected_item = self.tree.selection()
        
        if not selected_item:
            messagebox.showwarning("Warning", "Please select a student to delete.")
            return
        
        # Get student info from selected item
        student_info = self.tree.item(selected_item[0])['values']
        roll_no = student_info[0]  # Roll No is the first column
        
        # Confirm deletion
        if not messagebox.askyesno("Confirm Delete", 
                                f"Are you sure you want to delete student:\nRoll No: {roll_no}\nName: {student_info[1]}"):
            return
        
        # Delete from database
        session = SessionLocal()
        try:
            # First delete related attendance records
            session.query(Attendance).filter_by(roll_no=roll_no).delete()
            
            # Then delete the student
            student = session.query(Student).filter_by(roll_no=roll_no).first()
            if student:
                session.delete(student)
                session.commit()
                messagebox.showinfo("Success", "Student deleted successfully!")
                
                # Refresh the student list
                self.load_student_list()
            else:
                messagebox.showerror("Error", "Student not found in database.")
        
        except Exception as e:
            session.rollback()
            messagebox.showerror("Error", f"Failed to delete student: {str(e)}")
        
        finally:
            session.close()

    def create_subject_tab(self):
        """Create the Subjects management tab"""
        self.subject_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.subject_frame, text='Subjects')
        
        # Form frame
        form_frame = ttk.LabelFrame(self.subject_frame, text="Add/Edit Subject")
        form_frame.pack(fill='x', padx=10, pady=5)
        
        # Subject form fields
        ttk.Label(form_frame, text="Subject Name:").grid(row=0, column=0, padx=5, pady=5)
        self.subject_name_entry = ttk.Entry(form_frame)
        self.subject_name_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(form_frame, text="Subject Code:").grid(row=1, column=0, padx=5, pady=5)
        self.subject_code_entry = ttk.Entry(form_frame)
        self.subject_code_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(form_frame, text="Course:").grid(row=2, column=0, padx=5, pady=5)
        self.subject_course_entry = ttk.Combobox(form_frame, values=self.common_courses)
        self.subject_course_entry.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(form_frame, text="Batch:").grid(row=3, column=0, padx=5, pady=5)
        self.subject_batch_entry = ttk.Combobox(form_frame, values=self.batch_years)
        self.subject_batch_entry.grid(row=3, column=1, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(form_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Add Subject", command=self.add_subject).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Update Subject", command=self.update_subject).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Delete Subject", command=self.delete_subject).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Form", command=self.clear_subject_form).pack(side=tk.LEFT, padx=5)
        
        # Subject list
        list_frame = ttk.LabelFrame(self.subject_frame, text="Subject List")
        list_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Add search/filter options
        filter_frame = ttk.Frame(list_frame)
        filter_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(filter_frame, text="Filter by Course:").pack(side=tk.LEFT, padx=5)
        self.subject_filter_course = ttk.Combobox(filter_frame, values=['All'] + self.common_courses)
        self.subject_filter_course.set('All')
        self.subject_filter_course.pack(side=tk.LEFT, padx=5)
        self.subject_filter_course.bind('<<ComboboxSelected>>', self.filter_subjects)
        
        ttk.Label(filter_frame, text="Filter by Batch:").pack(side=tk.LEFT, padx=5)
        self.subject_filter_batch = ttk.Combobox(filter_frame, values=['All'] + self.batch_years)
        self.subject_filter_batch.set('All')
        self.subject_filter_batch.pack(side=tk.LEFT, padx=5)
        self.subject_filter_batch.bind('<<ComboboxSelected>>', self.filter_subjects)
        
        # Treeview for subjects
        self.subject_tree = ttk.Treeview(list_frame, 
                                        columns=('ID', 'Name', 'Code', 'Course', 'Batch', 'Created At'),
                                        show='headings')
        
        # Configure columns
        for col in self.subject_tree['columns']:
            self.subject_tree.heading(col, text=col)
            self.subject_tree.column(col, width=100)
        
        self.subject_tree.pack(fill='both', expand=True)
        self.subject_tree.bind('<<TreeviewSelect>>', self.on_subject_select)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.subject_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.subject_tree.configure(yscrollcommand=scrollbar.set)
        
        # Load subjects
        self.load_subject_list()

    def filter_subjects(self, event=None):
        """Filter subjects based on selected course and batch"""
        course_filter = self.subject_filter_course.get()
        batch_filter = self.subject_filter_batch.get()
        
        # Clear current list
        for item in self.subject_tree.get_children():
            self.subject_tree.delete(item)
        
        session = SessionLocal()
        try:
            # Start with base query
            query = session.query(Subject)
            
            # Apply filters if not 'All'
            if course_filter != 'All':
                query = query.filter(Subject.course == course_filter)
            if batch_filter != 'All':
                query = query.filter(Subject.batch == batch_filter)
            
            # Get filtered subjects
            subjects = query.all()
            
            # Populate treeview
            for subject in subjects:
                self.subject_tree.insert('', 'end', values=(
                    subject.id,
                    subject.name,
                    subject.code,
                    subject.course,
                    subject.batch,
                    subject.created_at
                ))
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to filter subjects: {str(e)}")
        finally:
            session.close()

    def add_subject(self):
        """Add a new subject to the database"""
        name = self.subject_name_entry.get()
        code = self.subject_code_entry.get()
        course = self.subject_course_entry.get()
        batch = self.subject_batch_entry.get()
        
        if not all([name, code, course, batch]):
            messagebox.showerror("Error", "All fields are required!")
            return
        
        session = SessionLocal()
        try:
            new_subject = Subject(
                id=f"SUB_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                name=name,
                code=code,
                course=course,
                batch=batch
            )
            session.add(new_subject)
            session.commit()
            messagebox.showinfo("Success", "Subject added successfully!")
            self.load_subject_list()
            self.clear_subject_form()
        except Exception as e:
            session.rollback()
            messagebox.showerror("Error", f"Failed to add subject: {str(e)}")
        finally:
            session.close()

    def clear_subject_form(self):
        """Clear the subject form fields"""
        self.subject_name_entry.delete(0, tk.END)
        self.subject_code_entry.delete(0, tk.END)
        self.subject_course_entry.set('')
        self.subject_batch_entry.set('')

    def on_subject_select(self, event):
        """Handle subject selection in treeview"""
        selected_item = self.subject_tree.selection()
        if selected_item:
            values = self.subject_tree.item(selected_item[0])['values']
            self.subject_name_entry.delete(0, tk.END)
            self.subject_name_entry.insert(0, values[1])
            self.subject_code_entry.delete(0, tk.END)
            self.subject_code_entry.insert(0, values[2])
            self.subject_course_entry.set(values[3])
            self.subject_batch_entry.set(values[4])

    def load_subject_list(self):
        """Load subjects into the treeview"""
        for item in self.subject_tree.get_children():
            self.subject_tree.delete(item)
        
        session = SessionLocal()
        try:
            subjects = session.query(Subject).all()
            for subject in subjects:
                self.subject_tree.insert('', 'end', values=(
                    subject.id,
                    subject.name,
                    subject.code,
                    subject.course,
                    subject.batch,
                    subject.created_at
                ))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load subject list: {str(e)}")
        finally:
            session.close()

    def update_subject(self):
        """Update an existing subject in the database"""
        selected_items = self.subject_tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Please select a subject to update!")
            return
        
        # Get form values
        name = self.subject_name_entry.get()
        code = self.subject_code_entry.get()
        course = self.subject_course_entry.get()
        batch = self.subject_batch_entry.get()
        
        if not all([name, code, course, batch]):
            messagebox.showerror("Error", "All fields are required!")
            return
        
        # Get subject ID from selected item
        subject_id = self.subject_tree.item(selected_items[0])['values'][0]
        
        session = SessionLocal()
        try:
            subject = session.query(Subject).filter_by(id=subject_id).first()
            if subject:
                # Update subject attributes
                subject.name = name
                subject.code = code
                subject.course = course
                subject.batch = batch
                
                session.commit()
                messagebox.showinfo("Success", "Subject updated successfully!")
                self.load_subject_list()
                self.clear_subject_form()
            else:
                messagebox.showerror("Error", "Subject not found!")
        except Exception as e:
            session.rollback()
            messagebox.showerror("Error", f"Failed to update subject: {str(e)}")
        finally:
            session.close()

    def delete_subject(self):
        """Delete the selected subject"""
        selected_items = self.subject_tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Please select a subject to delete!")
            return
        
        # Get subject info
        subject_values = self.subject_tree.item(selected_items[0])['values']
        subject_id = subject_values[0]
        subject_name = subject_values[1]
        
        # Confirm deletion
        if not messagebox.askyesno("Confirm Delete", 
                                f"Are you sure you want to delete subject:\n{subject_name}?"):
            return
        
        session = SessionLocal()
        try:
            # First check if there are any attendance records for this subject
            attendance_count = session.query(Attendance).filter_by(subject_id=subject_id).count()
            if attendance_count > 0:
                if not messagebox.askyesno("Warning", 
                    f"This subject has {attendance_count} attendance records. Deleting it will also delete all related attendance records. Continue?"):
                    return
                
                # Delete related attendance records
                session.query(Attendance).filter_by(subject_id=subject_id).delete()
            
            # Delete the subject
            session.query(Subject).filter_by(id=subject_id).delete()
            session.commit()
            
            messagebox.showinfo("Success", "Subject deleted successfully!")
            self.load_subject_list()
            self.clear_subject_form()
        
        except Exception as e:
            session.rollback()
            messagebox.showerror("Error", f"Failed to delete subject: {str(e)}")
        finally:
            session.close()

    def create_marks_tab(self):
        """Create the Marks management tab"""
        self.marks_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.marks_frame, text='Marks')
        
        # Form frame
        form_frame = ttk.LabelFrame(self.marks_frame, text="Add/Edit Marks")
        form_frame.pack(fill='x', padx=10, pady=5)
        
        # Student selection
        ttk.Label(form_frame, text="Student:").grid(row=0, column=0, padx=5, pady=5)
        self.marks_student_var = tk.StringVar()
        self.marks_student_combo = ttk.Combobox(form_frame, textvariable=self.marks_student_var)
        self.marks_student_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # Subject selection
        ttk.Label(form_frame, text="Subject:").grid(row=1, column=0, padx=5, pady=5)
        self.marks_subject_var = tk.StringVar()
        self.marks_subject_combo = ttk.Combobox(form_frame, textvariable=self.marks_subject_var)
        self.marks_subject_combo.grid(row=1, column=1, padx=5, pady=5)
        
        # Test Code
        ttk.Label(form_frame, text="Test Code:").grid(row=2, column=0, padx=5, pady=5)
        self.test_code_var = tk.StringVar()
        test_codes = ['CT1', 'CT2', 'CT3', 'MID', 'END']
        self.test_code_combo = ttk.Combobox(form_frame, textvariable=self.test_code_var, values=test_codes)
        self.test_code_combo.grid(row=2, column=1, padx=5, pady=5)
        
        # Marks
        ttk.Label(form_frame, text="Marks:").grid(row=3, column=0, padx=5, pady=5)
        self.marks_entry = ttk.Entry(form_frame)
        self.marks_entry.grid(row=3, column=1, padx=5, pady=5)
        
        # Buttons frame
        button_frame = ttk.Frame(form_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Submit Marks", command=self.upload_marks).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Form", command=self.clear_marks_form).pack(side=tk.LEFT, padx=5)
        
        # Marks list frame
        list_frame = ttk.LabelFrame(self.marks_frame, text="Marks List")
        list_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Filter frame
        filter_frame = ttk.Frame(list_frame)
        filter_frame.pack(fill='x', padx=5, pady=5)
        
        # Subject filter
        ttk.Label(filter_frame, text="Filter by Subject:").pack(side=tk.LEFT, padx=5)
        self.marks_filter_subject = ttk.Combobox(filter_frame)
        self.marks_filter_subject.pack(side=tk.LEFT, padx=5)
        
        # Test code filter
        ttk.Label(filter_frame, text="Filter by Test:").pack(side=tk.LEFT, padx=5)
        self.marks_filter_test = ttk.Combobox(filter_frame, values=['All'] + test_codes)
        self.marks_filter_test.pack(side=tk.LEFT, padx=5)
        self.marks_filter_test.set('All')
        
        # Filter button
        ttk.Button(filter_frame, text="Apply Filter", command=self.load_marks_list).pack(side=tk.LEFT, padx=5)
        
        # Marks treeview
        self.marks_tree = ttk.Treeview(list_frame, 
                                    columns=('Roll No', 'Name', 'Subject', 'Test Code', 'Marks'),
                                    show='headings')
        
        # Configure columns
        for col in self.marks_tree['columns']:
            self.marks_tree.heading(col, text=col)
            self.marks_tree.column(col, width=100)
        
        self.marks_tree.pack(fill='both', expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.marks_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.marks_tree.configure(yscrollcommand=scrollbar.set)
        
        # Initialize comboboxes
        self.update_marks_comboboxes()
        
        # Load initial marks list
        self.load_marks_list()

    def upload_marks(self):
        """Upload marks for a student"""
        # Get form values
        student = self.marks_student_var.get()
        subject = self.marks_subject_var.get()
        test_code = self.test_code_var.get()
        marks = self.marks_entry.get()
        
        # Validate inputs
        if not all([student, subject, test_code, marks]):
            messagebox.showerror("Error", "All fields are required!")
            return
        
        try:
            marks_value = float(marks)
            if marks_value < 0 or marks_value > 100:
                messagebox.showerror("Error", "Marks must be between 0 and 100!")
                return
        except ValueError:
            messagebox.showerror("Error", "Marks must be a valid number!")
            return
        
        # Extract roll number and subject code
        roll_no = student.split(' - ')[0]
        subject_code = subject.split(' - ')[0]
        
        session = SessionLocal()
        try:
            # Check if marks already exist for this combination
            existing_marks = session.query(Marks).filter_by(
                roll_no=roll_no,
                subject_code=subject_code,
                test_code=test_code
            ).first()
            
            if existing_marks:
                # Update existing marks
                existing_marks.marks = marks_value
                action = "updated"
            else:
                # Create new marks entry
                new_marks = Marks(
                    id=f"MRK_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    roll_no=roll_no,
                    subject_code=subject_code,
                    test_code=test_code,
                    marks=marks_value
                )
                session.add(new_marks)
                action = "added"
            
            session.commit()
            messagebox.showinfo("Success", f"Marks {action} successfully!")
            self.clear_marks_form()
            self.load_marks_list()
        
        except Exception as e:
            session.rollback()
            messagebox.showerror("Error", f"Failed to submit marks: {str(e)}")
        finally:
            session.close()

    def load_marks_list(self):
        """Load marks into the treeview with filtering"""
        # Clear existing items
        for item in self.marks_tree.get_children():
            self.marks_tree.delete(item)
        
        # Get filter values
        subject_filter = self.marks_filter_subject.get()
        test_filter = self.marks_filter_test.get()
        
        session = SessionLocal()
        try:
            # Base query with joins
            query = (
                session.query(Marks, Student, Subject)
                .join(Student, Marks.roll_no == Student.roll_no)
                .join(Subject, Marks.subject_code == Subject.code)
            )
            
            # Apply filters
            if subject_filter and subject_filter != 'All':
                subject_code = subject_filter.split(' - ')[0]
                query = query.filter(Marks.subject_code == subject_code)
            
            if test_filter and test_filter != 'All':
                query = query.filter(Marks.test_code == test_filter)
            
            # Execute query and populate treeview
            for marks, student, subject in query.all():
                self.marks_tree.insert('', 'end', values=(
                    student.roll_no,
                    student.name,
                    f"{subject.code} - {subject.name}",
                    marks.test_code,
                    f"{marks.marks:.2f}"
                ))
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load marks list: {str(e)}")
        finally:
            session.close()

    def update_marks_comboboxes(self):
        """Update the comboboxes in the marks tab with current data"""
        session = SessionLocal()
        try:
            # Update student combobox
            students = session.query(Student).all()
            self.marks_student_combo['values'] = [f"{s.roll_no} - {s.name}" for s in students]
            
            # Update subject combobox
            subjects = session.query(Subject).all()
            subject_values = [f"{s.code} - {s.name}" for s in subjects]
            self.marks_subject_combo['values'] = subject_values
            self.marks_filter_subject['values'] = ['All'] + subject_values
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update comboboxes: {str(e)}")
        finally:
            session.close()

    def clear_marks_form(self):
        """Clear all fields in the marks form"""
        self.marks_student_var.set('')
        self.marks_subject_var.set('')
        self.test_code_var.set('')
        self.marks_entry.delete(0, tk.END)
        

    def run(self):
        self.root.mainloop()

    def __del__(self):
        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    app = AttendanceSystem()
    app.run()