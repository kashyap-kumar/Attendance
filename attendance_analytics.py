import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, func, extract, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta

class AttendanceAnalytics:
    def __init__(self, engine):
        """
        Initialize analytics with database engine
        
        Args:
            engine: SQLAlchemy database engine
        """
        self.engine = engine
        self.Session = sessionmaker(bind=engine)

    def get_total_attendance_summary(self, days=30):
        """
        Generate overall attendance summary for past N days
        
        Args:
            days (int): Number of days to analyze
        
        Returns:
            dict: Summary of attendance metrics
        """
        session = self.Session()
        try:
            # Calculate date threshold
            threshold_date = datetime.now() - timedelta(days=days)
            
            # Total students
            total_students = session.execute(
                text("SELECT COUNT(*) FROM students")
            ).scalar()
            
            # Total attendance records
            total_attendance = session.execute(
                text("SELECT COUNT(*) FROM attendance WHERE date >= :threshold"), 
                {'threshold': threshold_date}
            ).scalar()
            
            # Unique students who attended
            unique_students = session.execute(
                text("SELECT COUNT(DISTINCT roll_no) FROM attendance WHERE date >= :threshold"), 
                {'threshold': threshold_date}
            ).scalar()
            
            # Average daily attendance
            daily_attendance = session.execute(
                text("""
                    SELECT 
                        DATE(date) as attendance_date, 
                        COUNT(DISTINCT roll_no) as daily_count 
                    FROM attendance 
                    WHERE date >= :threshold
                    GROUP BY DATE(date)
                """), 
                {'threshold': threshold_date}
            ).fetchall()
            
            avg_daily_attendance = (
                sum(row.daily_count for row in daily_attendance) / 
                len(daily_attendance) if daily_attendance else 0
            )
            
            return {
                'total_students': total_students,
                'total_attendance_records': total_attendance,
                'unique_students_attended': unique_students,
                'average_daily_attendance': round(avg_daily_attendance, 2),
                'attendance_percentage': round(
                    (unique_students / total_students) * 100, 2
                ) if total_students > 0 else 0
            }
        
        finally:
            session.close()

    def generate_attendance_trend(self, days=30):
        """
        Generate attendance trend for past N days
        
        Args:
            days (int): Number of days to analyze
        
        Returns:
            pandas.DataFrame: Daily attendance trend
        """
        session = self.Session()
        try:
            threshold_date = datetime.now() - timedelta(days=days)
            
            daily_attendance = session.execute(
                text("""
                    SELECT 
                        DATE(date) as attendance_date, 
                        COUNT(DISTINCT roll_no) as daily_count 
                    FROM attendance 
                    WHERE date >= :threshold
                    GROUP BY DATE(date)
                    ORDER BY attendance_date
                """), 
                {'threshold': threshold_date}
            ).fetchall()
            
            df = pd.DataFrame(
                daily_attendance, 
                columns=['date', 'attendance_count']
            )
            df['date'] = pd.to_datetime(df['date'])
            
            return df
        
        finally:
            session.close()

    def generate_student_attendance_report(self, days=30):
        """
        Generate individual student attendance report
        
        Args:
            days (int): Number of days to analyze
        
        Returns:
            pandas.DataFrame: Student-wise attendance report
        """
        session = self.Session()
        try:
            threshold_date = datetime.now() - timedelta(days=days)
            
            student_attendance = session.execute(
                text("""
                    SELECT 
                        s.roll_no, 
                        s.name, 
                        COUNT(a.id) as attendance_count,
                        ROUND(COUNT(a.id) * 100.0 / (
                            SELECT COUNT(DISTINCT DATE(date)) 
                            FROM attendance 
                            WHERE date >= :threshold
                        ), 2) as attendance_percentage
                    FROM students s
                    LEFT JOIN attendance a ON s.roll_no = a.roll_no 
                        AND a.date >= :threshold
                    GROUP BY s.roll_no, s.name
                    ORDER BY attendance_percentage DESC
                """), 
                {'threshold': threshold_date}
            ).fetchall()
            
            df = pd.DataFrame(
                student_attendance, 
                columns=[
                    'Roll No', 
                    'Name', 
                    'Attendance Count', 
                    'Attendance Percentage'
                ]
            )
            
            return df
        
        finally:
            session.close()

    def plot_attendance_trend(self, df):
        """
        Create a line plot of attendance trend
        
        Args:
            df (pandas.DataFrame): Daily attendance data
        
        Returns:
            matplotlib.figure.Figure: Attendance trend plot
        """
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['attendance_count'], marker='o')
        plt.title('Attendance Trend')
        plt.xlabel('Date')
        plt.ylabel('Number of Students Present')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()

    def generate_attendance_heatmap(self, df):
        """
        Create a heatmap of student attendance
        
        Args:
            df (pandas.DataFrame): Student attendance data
        
        Returns:
            matplotlib.figure.Figure: Attendance heatmap
        """
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            df[['Attendance Count', 'Attendance Percentage']].T, 
            annot=True, 
            cmap='YlGnBu',
            fmt='.2f'
        )
        plt.title('Student Attendance Heatmap')
        plt.tight_layout()
        return plt.gcf()