import argparse
import sys

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compare multiple videos side-by-side.")
    parser.add_argument("videos", nargs="+", help="Paths to the video files to compare.")
    parser.add_argument("--scale", type=float, default=0.5, help="Scaling factor for display to fit on screen.")
    parser.add_argument("--fps", type=int, default=30, help="Playback speed (FPS limit).")
    
    args = parser.parse_args()

    if len(args.videos) < 2:
        print("Please provide at least 2 videos to compare.")
        sys.exit(1)

    caps = [cv2.VideoCapture(vid) for vid in args.videos]

    # Verify all videos opened successfully
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"Error: Could not open video {args.videos[i]}")
            sys.exit(1)

    delay = int(1000 / args.fps)
    
    # Read the first frame to get dimensions
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print("Error: Could not read from one of the videos.")
            sys.exit(1)
            
    # Resize frames based on scale
    frames = [cv2.resize(f, (0, 0), fx=args.scale, fy=args.scale) for f in frames]
    
    # Calculate grid layout. If 2 videos, side-by-side. If 3 or 4, 2x2 grid.
    num_vids = len(caps)
    
    print(f"Comparing {num_vids} videos. Press 'q' to quit, 'p' to pause/resume.")
    
    paused = False

    while True:
        if not paused:
            frames = []
            all_ret = True
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    all_ret = False
                    break
                frames.append(cv2.resize(frame, (0, 0), fx=args.scale, fy=args.scale))
                
            if not all_ret:
                print("End of one or more videos reached.")
                break

            # Arrange frames vertically (one below the other)
            display = np.vstack(frames)

            # Add labels
            for i, vid_path in enumerate(args.videos):
                # Calculate Y position: top of each frame + offset
                y = i * frames[0].shape[0] + 25
                x = 10
                
                # Display full path in small font
                cv2.putText(display, vid_path, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # Adding a black outline for better readability
                cv2.putText(display, vid_path, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            cv2.imshow("Video Comparison", display)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
