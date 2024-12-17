import cv2
import os
import math

def extract_frames(video_path, output_dir, standard_frames):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")

    # Get video properties
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < standard_frames:
        raise ValueError(f"Video has only {total_frames} frames, fewer than the requested {standard_frames} frames.")

    # Calculate frame interval to get standard number of frames
    frame_interval = math.floor(total_frames / standard_frames)

    # Extract and save frames
    saved_frames = 0
    frame_index = 0
    while saved_frames < standard_frames:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = video_capture.read()

        if not success:
            break

        # Save the frame as an image
        frame_filename = os.path.join(output_dir, f"frame_{saved_frames + 1:03d}.jpg")
        cv2.imwrite(frame_filename, frame)

        saved_frames += 1
        frame_index += frame_interval

    video_capture.release()
    print(f"Extracted {saved_frames} frames and saved to {output_dir}")

video_path = r"C:\Users\suyash\IIITB\3D_vision_final_project\bottle_video.mp4"
output_dir = r"C:\Users\suyash\IIITB\3D_vision_final_project\images\bottle_mini"
standard_frames = 99  # number of frames we want

extract_frames(video_path, output_dir, standard_frames)
