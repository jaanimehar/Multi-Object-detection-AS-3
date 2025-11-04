====================================================================
PROJECT: MULTI-OBJECT TRACKING (MOT) USING SORT AND MOTMETRICS
====================================================================

AUTHOR: Meharban Saifi
====================================================================

📘 DESCRIPTION:
---------------
This project implements a complete end-to-end Multi-Object Tracking (MOT) pipeline
using the SORT algorithm and evaluates tracking accuracy with the MOT metrics library.

The system reads an image sequence, detects and tracks multiple moving objects,
generates a bounding-box video output, saves tracking results, and computes key metrics
such as MOTA (Accuracy) and MOTP (Precision).

====================================================================
📂 PROJECT STRUCTURE:
====================================================================

│
├── img1/                      → Folder containing all input image frames (.jpg)
│     ├── 000001.jpg
│     ├── 000002.jpg
│     └── ...
│
├── gt.txt                     → Ground truth data file
├── moto_track.py              → Main Python file (tracking + evaluation)
├── tracking_results.txt       → Auto-generated tracking output
├── output_video.mp4           → Video with bounding boxes drawn on tracked objects
├── mot_metrics_summary.txt    → Evaluation summary of tracking results
└── README.txt                 → This documentation file

====================================================================
⚙️ SETUP & EXECUTION STEPS:
====================================================================

1️⃣ INSTALL REQUIRED LIBRARIES
------------------------------
Ensure Python 3.8+ is installed, then install dependencies using:

```bash
pip install numpy opencv-python motmetrics

2️⃣ PREPARE THE DATASET

Place all image frames in a folder named img1 (as provided by your professor).
Ensure gt.txt (ground truth file) is in the same project directory.

File structure example:

D:\M.Tech DS\AIML application\Multi Object detection AS-3\
│
├── moto_track.py
├── gt.txt
├── img1\
│     ├── 000001.jpg
│     ├── 000002.jpg
│     └── ...

3️⃣ RUN THE MAIN SCRIPT

Execute the single file that handles both tracking and evaluation:

python moto_track.py


This will:

Read frames from img1/

Perform object detection and tracking using SORT

Draw bounding boxes and object IDs on each frame

Save tracking results to tracking_results.txt

Generate an output video output_video.mp4

Evaluate performance using gt.txt

Save evaluation summary to mot_metrics_summary.txt

4️⃣ VIEW RESULTS

Once processing completes, you’ll see console output similar to:

Processing 145 frames...
Processed 50/145 frames
Processed 100/145 frames
Video saved to: D:\M.Tech DS\AIML application\Multi Object detection AS-3\output_video.mp4
Tracking results saved to: D:\M.Tech DS\AIML application\Multi Object detection AS-3\tracking_results.txt
Evaluating tracking performance...

============================================================
MOT Evaluation Results
============================================================
        mota      motp  num_frames  num_matches  num_false_positives  num_misses  num_switches
MOT  0.62532  0.778437         145          489                    0         284             9
============================================================

Key Metrics:
MOTA (Multiple Object Tracking Accuracy): 62.53%
MOTP (Multiple Object Tracking Precision): 0.7784

====================================================================
📊 OUTPUT FILES EXPLAINED:
File Name	Description
output_video.mp4	Video showing object tracking with bounding boxes
tracking_results.txt	Contains [frame, id, x, y, w, h] for each detection
mot_metrics_summary.txt	Contains computed MOT metrics summary
gt.txt	Ground truth for evaluation
====================================================================
✅ FINAL PERFORMANCE:

Frames Processed: 145

MOTA (Accuracy): 62.53%

MOTP (Precision): 0.7784

False Positives: 0

Misses: 284

Switches: 9

====================================================================
🧠 NOTES:

Ensure that the image filenames are continuous (000001.jpg, 000002.jpg, …)

Both tracking and evaluation are performed automatically by moto_track.py

You can improve tracking accuracy by adjusting SORT parameters such as:
max_age, min_hits, and iou_threshold.

====================================================================
🔍 Interpretation & Analysis

✅ 1. MOTA = 62.53%
This value represents the overall tracking accuracy — how well the system detects and maintains identities over all frames.
A 62.53% MOTA indicates moderate tracking accuracy.
It means the tracker correctly identified and followed most objects, though some were missed or mismatched due to occlusions or rapid motion.

✅ 2. MOTP = 0.7784
This measures how precisely the predicted bounding boxes overlap with the ground truth.
A value close to 1.0 means very accurate localization.
Thus, 0.7784 suggests the bounding boxes are fairly accurate and well-aligned with actual object positions.

✅ 3. False Positives = 0
This is excellent — it shows the tracker did not produce spurious detections. Every tracked object likely corresponded to a real target.

✅ 4. Misses = 284
A relatively high number of missed detections implies that some objects were not consistently detected across frames, possibly due to lighting, occlusion, or partial visibility.

✅ 5. ID Switches = 9
This means object identities were swapped 9 times during tracking.
This is within an acceptable range, showing that the tracker mostly maintained consistent IDs, though improvements in motion modeling could reduce switches further.

📊 Overall Comment

The tracker performs reasonably well, achieving:

High precision (MOTP ~0.78)

Moderate accuracy (MOTA ~62%)

Very few ID switches and zero false positives

However, the miss rate suggests potential improvement areas — better detection sensitivity or parameter tuning (e.g., lower iou_threshold, higher detection confidence) could improve tracking completeness.
