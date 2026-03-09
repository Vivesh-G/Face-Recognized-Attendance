import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
from PIL import Image

# Add parent directory to path to import AttendanceSystem
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import AttendanceSystem

# Page Configuration
st.set_page_config(page_title="Face Attendance System", page_icon="👤", layout="wide")

# Initialize Attendance System (Cached)
@st.cache_resource
def get_system():
    # Use the local checkpoint path
    return AttendanceSystem(model_path="../lafs_webface_finetune_withaugmentation.pth")

system = get_system()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Enrollment", "Mark Attendance", "View Logs"])

# Home Page
if page == "Home":
    st.title("Landmark-Aware Face Attendance System")
    st.markdown("""
    Welcome to the Part-fViT based Face Recognition Attendance System.
    
    ### System Overview:
    - **Model:** Part-fViT (Landmark-aware Vision Transformer)
    - **Detection:** MTCNN
    - **Features:** 
        - Enroll new students with a simple photo.
        - Mark attendance using live snapshots.
        - Export and view attendance logs in CSV format.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Registered Students", len(system.known_embeddings))
    with col2:
        if os.path.exists('../data/attendance.csv'):
            df = pd.read_csv('../data/attendance.csv')
            st.metric("Total Records", len(df))

# Enrollment Page
elif page == "Enrollment":
    st.title("Student Enrollment")
    
    name = st.text_input("Enter Student Name")
    enroll_method = st.radio("Enrollment Method", ["Camera", "Upload Image"])
    
    if enroll_method == "Camera":
        img_file = st.camera_input("Take a photo of the student")
    else:
        img_file = st.file_uploader("Upload student photo", type=['jpg', 'jpeg', 'png'])
        
    if img_file and name:
        if st.button("Register Student"):
            # Convert to OpenCV format
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            success, message = system.register_student(name, img)
            if success:
                st.success(message)
                # Save physical copy
                os.makedirs('../data/students', exist_ok=True)
                cv2.imwrite(f'../data/students/{name}.jpg', img)
            else:
                st.error(message)

# Attendance Page
elif page == "Mark Attendance":
    st.title("Mark Attendance")
    st.write("Position your face in the camera and take a snapshot.")
    
    img_file = st.camera_input("Attendance Snapshot")
    threshold = st.slider("Recognition Threshold", 0.0, 1.0, 0.6)
    
    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        
        # Detect faces
        boxes, _ = system.face_detector.detect(frame)
        
        if boxes is not None:
            found_any = False
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                face_crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
                
                if face_crop.size == 0: continue
                
                # Get embedding
                live_emb = system.get_embedding(face_crop)
                
                best_score = -1
                best_name = "Unknown"
                
                for name, ref_emb in system.known_embeddings.items():
                    score = np.dot(live_emb, ref_emb)
                    if score > best_score:
                        best_score = score
                        best_name = name
                
                if best_score > threshold:
                    if system.mark_attendance(best_name):
                        st.success(f"Verified: **{best_name}** (Score: {best_score:.2f})")
                        system.export_attendance('../data/attendance.csv')
                    else:
                        st.info(f"Already marked: **{best_name}**")
                    found_any = True
                else:
                    st.warning(f"Face detected but not recognized (Best match: {best_name}, Score: {best_score:.2f})")
            
            if not found_any:
                st.info("No registered faces recognized.")
        else:
            st.error("No faces detected in the image.")

# Logs Page
elif page == "View Logs":
    st.title("Attendance Logs")
    
    if os.path.exists('../data/attendance.csv'):
        df = pd.read_csv('../data/attendance.csv')
        st.dataframe(df, use_container_width=True)
        
        # CSV Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            csv,
            "attendance.csv",
            "text/csv",
            key='download-csv'
        )
        
        if st.button("Clear Session Logs"):
            system.attendance_log = []
            st.rerun()
    else:
        st.info("No attendance records found yet.")
