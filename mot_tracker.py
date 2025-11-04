"""
Multi-Object Tracking with YOLO Detection and SORT Algorithm
Assignment-3: AI/ML and Applications (MDS 302)
"""

import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import glob
import os
from pathlib import Path

# ============================================================================
# SORT IMPLEMENTATION
# ============================================================================

def linear_assignment(cost_matrix):
    """Hungarian algorithm for linear assignment"""
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    Compute IOU between two bounding boxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
    return(o)  


def convert_bbox_to_z(bbox):
    """
    Convert bounding box [x1,y1,x2,y2] to [x,y,s,r]
    where x,y is the center, s is scale/area, r is aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Convert [x,y,s,r] to bounding box [x1,y1,x2,y2]
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1,5))


class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    
    def __init__(self, bbox):
        """Initialize a tracker using initial bounding box"""
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],  
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """Update the state with observed bbox"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """Advance the state and return the predicted bounding box estimate"""
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """Return the current bounding box estimate"""
        return convert_x_to_bbox(self.kf.x)


class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Parameters:
          max_age: Maximum number of frames to keep alive a track without associated detections.
          min_hits: Minimum number of associated detections before track is confirmed.
          iou_threshold: Minimum IOU for match.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Parameters:
          dets: numpy array of detections in format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Returns:
          numpy array of tracked objects in format [[x1,y1,x2,y2,id],[x1,y1,x2,y2,id],...]
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
        
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1))
            i -= 1
            
            # Remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,5))


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    # Filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# ============================================================================
# YOLO DETECTION AND TRACKING PIPELINE
# ============================================================================

class MOTTracker:
    def __init__(self, sequence_path, output_path, detection_confidence=0.5):
        """
        Initialize the Multi-Object Tracker
        
        Parameters:
            sequence_path: Path to MOT15 sequence folder (contains 'img1' folder)
            output_path: Path to save output video and tracking results
            detection_confidence: Confidence threshold for YOLO detections
        """
        self.sequence_path = Path(sequence_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.detection_confidence = detection_confidence
        
        # Initialize SORT tracker
        self.tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
        
        # Load YOLO model (using YOLOv8 from ultralytics)
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')  # Use nano model for speed
            self.use_yolo = True
        except ImportError:
            print("Warning: ultralytics not installed. Using dummy detections.")
            self.use_yolo = False
        
        # Get image paths
        self.img_folder = self.sequence_path / 'img1'
        self.img_paths = sorted(glob.glob(str(self.img_folder / '*.jpg')))
        
        if not self.img_paths:
            self.img_paths = sorted(glob.glob(str(self.img_folder / '*.png')))
        
        # Tracking results storage
        self.tracking_results = []
        
        # Color palette for visualization
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(200, 3), dtype=np.uint8)

    def detect_objects(self, frame):
        """
        Run YOLO detection on frame
        Returns: detections in format [[x1,y1,x2,y2,score],...]
        """
        if not self.use_yolo:
            # Return empty detections if YOLO not available
            return np.empty((0, 5))
        
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        # Filter for person class (class 0 in COCO dataset)
        for box in results.boxes:
            if box.cls[0] == 0 and box.conf[0] > self.detection_confidence:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detections.append([x1, y1, x2, y2, conf])
        
        return np.array(detections) if detections else np.empty((0, 5))

    def draw_tracks(self, frame, tracked_objects):
        """Draw bounding boxes and IDs on frame"""
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)
            
            # Get color for this ID
            color = tuple(map(int, self.colors[track_id % len(self.colors)]))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID
            label = f'ID: {track_id}'
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

    def run_tracking(self):
        """Run tracking on entire sequence"""
        print(f"Processing {len(self.img_paths)} frames...")
        
        # Get video properties
        first_frame = cv2.imread(self.img_paths[0])
        height, width = first_frame.shape[:2]
        
        # Initialize video writer
        output_video = self.output_path / 'output_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video), fourcc, 20.0, (width, height))
        
        for frame_idx, img_path in enumerate(self.img_paths, start=1):
            frame = cv2.imread(img_path)
            
            # Detect objects
            detections = self.detect_objects(frame)
            
            # Update tracker
            tracked_objects = self.tracker.update(detections)
            
            # Store results in MOT format: frame, id, x, y, w, h, conf, -1, -1, -1
            for track in tracked_objects:
                x1, y1, x2, y2, track_id = track
                w, h = x2 - x1, y2 - y1
                self.tracking_results.append([
                    frame_idx, int(track_id), x1, y1, w, h, 1, -1, -1, -1
                ])
            
            # Draw tracks
            frame = self.draw_tracks(frame, tracked_objects)
            
            # Write frame
            out.write(frame)
            
            if frame_idx % 50 == 0:
                print(f"Processed {frame_idx}/{len(self.img_paths)} frames")
        
        out.release()
        print(f"Video saved to: {output_video}")
        
        # Save tracking results
        self.save_results()

    def save_results(self):
        """Save tracking results in MOT format"""
        output_file = self.output_path / 'tracking_results.txt'
        
        with open(output_file, 'w') as f:
            for result in self.tracking_results:
                line = ','.join(map(str, result))
                f.write(line + '\n')
        
        print(f"Tracking results saved to: {output_file}")
         


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_mot(gt_file, result_file):
    """
    Evaluate MOT metrics using motmetrics
    
    Parameters:
        gt_file: Path to ground truth file (gt.txt)
        result_file: Path to tracking results file
    """
    try:
        import motmetrics as mm
    except ImportError:
        print("Please install motmetrics: pip install motmetrics")
        return
    
    # Load ground truth
    gt = np.loadtxt(gt_file, delimiter=',')
    
    # Load tracking results
    results = np.loadtxt(result_file, delimiter=',')
    
    # Create accumulator
    acc = mm.MOTAccumulator(auto_id=True)
    
    # Get unique frames
    frames_gt = np.unique(gt[:, 0]).astype(int)
    frames_res = np.unique(results[:, 0]).astype(int)
    all_frames = sorted(set(frames_gt) | set(frames_res))
    
    print("Evaluating tracking performance...")
    
    for frame in all_frames:
        # Get ground truth for this frame
        gt_frame = gt[gt[:, 0] == frame]
        gt_ids = gt_frame[:, 1].astype(int)
        gt_boxes = gt_frame[:, 2:6]  # x, y, w, h
        
        # Get results for this frame
        res_frame = results[results[:, 0] == frame]
        res_ids = res_frame[:, 1].astype(int)
        res_boxes = res_frame[:, 2:6]  # x, y, w, h
        
        # Convert to [x1, y1, x2, y2] format
        gt_boxes_xyxy = np.column_stack([
            gt_boxes[:, 0],
            gt_boxes[:, 1],
            gt_boxes[:, 0] + gt_boxes[:, 2],
            gt_boxes[:, 1] + gt_boxes[:, 3]
        ])
        
        res_boxes_xyxy = np.column_stack([
            res_boxes[:, 0],
            res_boxes[:, 1],
            res_boxes[:, 0] + res_boxes[:, 2],
            res_boxes[:, 1] + res_boxes[:, 3]
        ])
        
        # Calculate distances (1 - IOU)
        if len(gt_boxes_xyxy) > 0 and len(res_boxes_xyxy) > 0:
            iou_matrix = iou_batch(res_boxes_xyxy, gt_boxes_xyxy)
            distances = 1 - iou_matrix
        else:
            distances = np.empty((0, 0))
        
        # Update accumulator
        acc.update(
            gt_ids,
            res_ids,
            distances
        )
    
    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['mota', 'motp', 'num_frames', 'num_matches', 
                                       'num_false_positives', 'num_misses', 
                                       'num_switches'], name='MOT')
    
    print("\n" + "="*60)
    print("MOT Evaluation Results")
    print("="*60)
    print(summary)
    print("="*60)
    
    # Extract key metrics
    mota = summary['mota'].values[0] * 100
    motp = summary['motp'].values[0]
    
    print(f"\nKey Metrics:")
    print(f"MOTA (Multiple Object Tracking Accuracy): {mota:.2f}%")
    print(f"MOTP (Multiple Object Tracking Precision): {motp:.4f}")
 
    return summary



# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
       # Path to MOT15 sequence
       sequence_path = "D:\M.Tech DS\AIML application\Multi Object detection AS-3"
       output_path = "D:\M.Tech DS\AIML application\Multi Object detection AS-3"
       
       # Initialize and run tracker
       tracker = MOTTracker(
           sequence_path=sequence_path,
           output_path=output_path,
           detection_confidence=0.5
       )
       
       tracker.run_tracking()
       
       # Evaluate results
       gt_file = f"{sequence_path}/gt.txt"
       result_file = f"{output_path}/tracking_results.txt"
       
       if os.path.exists(gt_file):
           evaluate_mot(gt_file, result_file)
       else:
           print(f"Ground truth file not found: {gt_file}")


