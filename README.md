# Object Detection and Tracking (Traffic Speed Enforcement)

## Overview
This repository contains a professional-grade traffic speed enforcement application. It utilizes state-of-the-art YOLO-based object detection and tracking to monitor vehicular traffic, calculate estimated speeds across predefined geographical thresholds, and issue virtual e-challans for vehicles exceeding the specified speed limits.

The system incorporates robust multi-vehicle tracking, memory leak prevention via stale track eviction, and automated summary analytics reporting. It is wrapped in a Streamlit web interface designed for high-definition video processing and data visualization.

## Key Features
* **YOLOv8 Inference**: Real-time bounding box detection for specific vehicle classes (cars, motorcycles, buses, trucks).
* **Speed Calculation Algorithm**: Computes vehicle speed by accurately measuring the time offset between crossing distinct entry and exit demarcations.
* **Stale Track Eviction**: Proactively handles occlusions and edge-case exits to maintain memory efficiency and tracking integrity.
* **HD Video Export**: Automatically synthesizes and exports a high-definition (HD) MP4 video evidence file containing boundary markers, real-time timestamps, and compliance status tags.
* **Automated Analytics**: Generates extensive CSV reports alongside statistical distributions plotted utilizing Seaborn and Matplotlib.
* **Streamlit Interface**: Integrated frontend for configuration, file uploading, and playback validation.

## Application Architecture
The core logic relies on `ultralytics` for inference and tracking, while standard `OpenCV` logic performs frame-level annotation and speed delta measurements. The analytical engine processes the tracking dictionary data structures, cross-referencing pixel deltas against manually defined scaling constants.

### Configuration Constants
The accuracy of the speed tracking depends significantly on real-world calibration variables:
- `REAL_DIST_METERS`: The physical geographic distance between Line A and Line B.
- `SPEED_LIMIT_KMH`: The defined threshold triggering excessive speed violations.
- `LINE_A_RATIO` and `LINE_B_RATIO`: Proportional height offsets reflecting the calibration zones on the 2D video plane.

## Installation and Deployment

### Requirements
The software executes via Python 3.9+ environments natively or within Dockerized containers. Ensure hardware acceleration capability for real-time processing performance. 

Installation commands:
```bash
pip install -r requirements.txt
```

### Execution
To initiate the internal dashboard on a local server instance, execute the Streamlit runtime:
```bash
streamlit run app.py
```
Upload instances of traffic feed MP4 files to process and review analytics immediately within the interactive session.

## Output Assets
Upon successfully parsing a video file, the system provides:
1. `echallan_evidence.mp4`: High-definition video with HUD indicators emphasizing violators.
2. `violations_report.csv`: Time-series log containing violation matrices.
3. `speed_analysis_report.png`: Graphical synthesis of general traffic performance trends versus legal limits.

## Notice
This system models estimated velocities and is designed for prototype evaluation and statistical analysis. Professional deployment requires static camera mounting, defined perspective transformations, and external calibration validation for legal adherence.
