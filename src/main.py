import sys
import os
import cv2
from pose_2d import run_2D_extraction
from pose_3d import run_3D_extraction
from smpl_parameters import calculate_smpl
import argparse
import torch
import gc

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
output_folder = os.path.join(base_dir, "data", "output_folder")
input_folder = os.path.join(base_dir, "data", "input_data")

def clear_memory():
    """Function to clear CPU and GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def record_video(video_path):
    """Function to record a video using the webcam and save it."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))
    
    print("Recording... Press 'q' to stop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow('Recording', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved at: {video_path}")
    return video_path

def main():
    parser = argparse.ArgumentParser(description="Pose Estimation Script")
    parser.add_argument('--video_path', type=str, help='Path to the input video file')
    parser.add_argument('--output_path', type=str, help='Output path for smplx and .bvh files.', default=output_folder)
    parser.add_argument('--record', action='store_true', help='Record a video using the webcam')
    
    args = parser.parse_args()
    
    if args.record:
        video_path = os.path.join(input_folder, "recorded_video.avi")
        video_path = record_video(video_path)
    else:
        if not args.video_path:
            print("Error: You must provide a video path or use --record to capture a video.")
            sys.exit(1)
        video_path = args.video_path
    
    output_path = args.output_path
    
    print(f"Video Path: {video_path}")
    print(f"Output Path: {output_path}")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    temp_folder = os.path.join(base_dir, "data", "temp")
    
    # Run 2D extraction
    run_2D_extraction(video_path, temp_folder)
    clear_memory()
    
    # Run 3D extraction
    run_3D_extraction(video_path)
    
    # Calculate SMPL
    calculate_smpl(output_folder)

if __name__ == "__main__":
    main()

#python main.py --video_path "E:\Users\Philipp\Dokumente\Pose_Estimation_3D\Alphapose\AlphaPose\videos\cxk.mp4"sd