"""Minimal API example for using PlayerTracker programmatically."""

import sys
from pathlib import Path

import cv2

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.player_tracking import PlayerTracker


def main():
    # Initialize the tracker with default weights (yolov8n.pt)
    tracker = PlayerTracker()

    # Open your video
    video_path = "data/sample_match.mp4.webm"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    # Track frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get YOLOv8 results object
        results = tracker.track_frame(frame, persist=True)

        # You can access raw bounding boxes and tracking IDs:
        if results.boxes is not None and results.boxes.id is not None:
            ids = results.boxes.id.cpu().numpy().astype(int)
            print(f"Detected {len(ids)} people with IDs: {ids}")

        # Or just use ultralytics built-in plotting:
        annotated_frame = results.plot()
        cv2.imshow("Raw YOLO Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
