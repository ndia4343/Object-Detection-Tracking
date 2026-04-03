import streamlit as st
import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from ultralytics import YOLO
import os
import tempfile
import subprocess

st.set_page_config(page_title="Traffic Speed Enforcement Tracker", layout="wide")

st.title("Traffic Speed Enforcement Tracker")
st.markdown("Upload a traffic feed video for object tracking, velocity estimation, and automated e-challan generation.")

# Parameters Configuration
with st.sidebar:
    st.header("Calibration Settings")
    real_dist_meters = st.number_input("Real Distance Between Lines (m)", value=15.0, step=1.0)
    line_a_ratio = st.slider("Entry Line Offset (Ratio)", 0.0, 1.0, 0.40)
    line_b_ratio = st.slider("Challan Line Offset (Ratio)", 0.0, 1.0, 0.70)
    speed_limit_kmh = st.number_input("Speed Limit Threshold (km/h)", value=60)
    conf_threshold = st.slider("Object Detection Confidence", 0.1, 1.0, 0.40)

uploaded_file = st.file_uploader("Upload Video File (MP4 format optimally)", type=["mp4", "avi", "mov"])

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

if uploaded_file is not None:
    st.info("File verification successful. Proceeding to initialize deep learning tracking protocol.")
    
    if st.button("Initialize Processing"):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        # Process video writing dynamically to save RAM for 2GB+ files
        uploaded_file.seek(0)
        while True:
            chunk = uploaded_file.read(10 * 1024 * 1024)  # 10MB chunks
            if not chunk:
                break
            tfile.write(chunk)
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("Failed to process the uploaded video feed format.")
        else:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            line_a = int(h * line_a_ratio)
            line_b = int(h * line_b_ratio)
            
            # Temporary internal processing file
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (w, h))

            vehicle_classes = [2, 3, 5, 7]
            max_tracking_age_frames = 300
            
            entry_times = {}
            entry_frame_ts = {}
            violation_log = []
            
            video_start_time = datetime.now()
            
            st.write("Executing analysis pass across frames...")
            progress_bar = st.progress(0)
            
            current_frame = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                current_frame += 1
                progress_bar.progress(min(current_frame / float(total_frames), 1.0))
                
                results = model.track(
                    frame,
                    persist=True,
                    conf=conf_threshold,
                    classes=vehicle_classes,
                    verbose=False
                )
                
                annotated = results[0].plot()
                
                cv2.line(annotated, (0, line_a), (w, line_a), (0, 220, 220), 2)
                cv2.line(annotated, (0, line_b), (w, line_b), (0, 0, 255), 3)
                
                cv2.putText(annotated, "ENTRY LINE", (8, line_a - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(annotated, "CHALLAN LINE", (8, line_b - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                
                frame_time = video_start_time + timedelta(seconds=current_frame / fps)
                ts_str = frame_time.strftime('%Y-%m-%d %H:%M:%S')
                cv2.putText(annotated, ts_str, (w - 300, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    ids = results[0].boxes.id.int().cpu().tolist()
                    
                    stale_ids = [vid for vid, e_frame in entry_frame_ts.items() if current_frame - e_frame > max_tracking_age_frames]
                    for vid in stale_ids:
                        entry_times.pop(vid, None)
                        entry_frame_ts.pop(vid, None)
                        
                    for box, obj_id in zip(boxes, ids):
                        x1, y1, x2, y2 = box
                        y_bottom = y2
                        
                        if y_bottom > line_a and obj_id not in entry_times:
                            entry_times[obj_id] = current_frame
                            entry_frame_ts[obj_id] = current_frame
                            
                        if y_bottom > line_b and obj_id in entry_times:
                            start_frame = entry_times.pop(obj_id)
                            entry_frame_ts.pop(obj_id, None)
                            
                            frame_diff = current_frame - start_frame
                            time_seconds = frame_diff / fps
                            
                            if time_seconds > 0 and frame_diff >= 2:
                                speed_kmh = (real_dist_meters / time_seconds) * 3.6
                                
                                if speed_kmh <= 300 and speed_kmh > speed_limit_kmh:
                                    challan_time = (video_start_time + timedelta(seconds=current_frame / fps)).strftime('%Y-%m-%d %H:%M:%S')
                                    violation_log.append({
                                        'Vehicle_ID': obj_id,
                                        'Speed_KMH': round(speed_kmh, 2),
                                        'Speed_Limit_KMH': speed_limit_kmh,
                                        'Excess_KMH': round(speed_kmh - speed_limit_kmh, 2),
                                        'Frame': current_frame,
                                        'Timestamp': challan_time,
                                        'Status': 'EXCESSIVE SPEED'
                                    })
                                    
                                    label = f"CHALLAN! {round(speed_kmh, 1)} km/h"
                                    cv2.rectangle(annotated, (int(x1), int(y1) - 26), (int(x1) + len(label) * 11, int(y1)), (0, 0, 200), -1)
                                    cv2.putText(annotated, label, (int(x1) + 3, int(y1) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                                    
                out.write(annotated)
                
            cap.release()
            out.release()
            progress_bar.empty()
            
            st.success("Frame processing sequence completed.")
            
            with st.spinner("Transcoding video for internal HTML5 integration (HD deployment format)..."):
                final_output = "echallan_evidence.mp4"
                command = ["ffmpeg", "-y", "-i", temp_output, "-vcodec", "libx264", final_output]
                try:
                    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except FileNotFoundError:
                    st.warning("FFmpeg executable not identified within system path variables. Reverting to structural mp4v encoding. Playback may not render correctly in standard browsers.")
                    final_output = temp_output
                except Exception as e:
                    st.warning(f"Error during video format transcription: {e}")
                    final_output = temp_output

            st.header("Processing Resolution")
            df_violations = pd.DataFrame(violation_log)
            if not df_violations.empty:
                st.write("E-Challan Issuance Log")
                st.dataframe(df_violations)
                
                sns.set_theme(style="whitegrid")
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                sns.histplot(df_violations['Speed_KMH'], kde=True, color='crimson', bins=15, ax=axes[0])
                axes[0].axvline(speed_limit_kmh, color='royalblue', linestyle='--', linewidth=2, label=f'Speed Limit ({speed_limit_kmh} km/h)')
                axes[0].set_title('Speed Violation Distribution', fontsize=14)
                axes[0].set_xlabel('Speed (km/h)')
                axes[0].set_ylabel('Number of Vehicles')
                axes[0].legend()
                
                sns.histplot(df_violations['Excess_KMH'], kde=True, color='darkorange', bins=15, ax=axes[1])
                axes[1].set_title('Excess Speed Over Threshold limit', fontsize=14)
                axes[1].set_xlabel('Excess km/h over limit')
                axes[1].set_ylabel('Number of Vehicles')
                
                plt.suptitle('Enforcement Analytics Synthesis', fontsize=16, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Zero violations detected across evaluated parameters.")
            
            st.video(final_output)

            # Cleanup resources
            if os.path.exists(tfile.name):
                os.remove(tfile.name)
