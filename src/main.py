import cv2
import pandas as pd
from datetime import datetime, timedelta
from ultralytics import YOLO
import os

# --- Configuration Constants (Calibrated for YouTube Demo) ---
REAL_DIST_METERS = 15.0  # Physical distance between Entry and Challan lines
LINE_A_RATIO = 0.40      # Yellow Entry Line position (0.0 to 1.0)
LINE_B_RATIO = 0.70      # Red Challan Line position (0.0 to 1.0)
SPEED_LIMIT_KMH = 60.0   # Trigger for e-challan
CONF_THRESHOLD = 0.40    # Model confidence threshold
VEHICLE_CLASSES = [2, 3, 5, 7] # COCO: car, motorcycle, bus, truck

# --- Path Configuration ---
MODEL_PATH = "../models/yolov8n.pt"
INPUT_VIDEO = "../data/input/traffic_sample.mp4"
OUTPUT_VIDEO = "../data/output/echallan_evidence.mp4"
OUTPUT_CSV = "../data/output/violations_report.csv"

def run_engine():
    # Load Model
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error: Could not open video at {INPUT_VIDEO}")
        return

    # Video Metadata
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    line_a, line_b = int(h * LINE_A_RATIO), int(h * LINE_B_RATIO)
    
    # Initialize Writer
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # Tracking Variables
    entry_times = {}
    violation_log = []
    video_start_time = datetime.now()
    current_frame = 0

    print("Processing started. Extracting intelligence from frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        current_frame += 1

        # YOLO Tracking Inference
        results = model.track(frame, persist=True, conf=CONF_THRESHOLD, classes=VEHICLE_CLASSES, verbose=False)
        annotated = results[0].plot()

        # HUD Overlay
        cv2.line(annotated, (0, line_a), (w, line_a), (0, 220, 220), 2) # Entry
        cv2.line(annotated, (0, line_b), (w, line_b), (0, 0, 255), 3)   # Challan
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().tolist()

            for box, obj_id in zip(boxes, ids):
                y_bottom = box[3]

                # Logic: Entry Line Crossing
                if y_bottom > line_a and obj_id not in entry_times:
                    entry_times[obj_id] = current_frame

                # Logic: Challan Line Crossing & Speed Calculation
                if y_bottom > line_b and obj_id in entry_times:
                    start_frame = entry_times.pop(obj_id)
                    time_seconds = (current_frame - start_frame) / fps
                    
                    if time_seconds > 0:
                        speed_kmh = (REAL_DIST_METERS / time_seconds) * 3.6
                        
                        if speed_kmh > SPEED_LIMIT_KMH and speed_kmh < 300:
                            timestamp = (video_start_time + timedelta(seconds=current_frame/fps)).strftime('%H:%M:%S')
                            violation_log.append({
                                'Vehicle_ID': obj_id,
                                'Speed_KMH': round(speed_kmh, 2),
                                'Timestamp': timestamp
                            })
                            # Visual Alert on Frame
                            cv2.putText(annotated, f"CHALLAN! {round(speed_kmh,1)}km/h", (int(box[0]), int(box[1]-10)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out.write(annotated)

    # Save Analytical Report
    pd.DataFrame(violation_log).to_csv(OUTPUT_CSV, index=False)
    
    cap.release()
    out.release()
    print(f"Processing Complete.\nVideo saved: {OUTPUT_VIDEO}\nReport saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_engine()
                    
