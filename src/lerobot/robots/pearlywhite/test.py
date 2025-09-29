# from lerobot.datasets.lerobot_dataset import LeRobotDataset
# import numpy as np

# dataset = LeRobotDataset(repo_id="rdteteam/hello10_from_azure")
# print(len(dataset[0]['observation.images.cam_1']))

# for frame in dataset:
#     np_frame = np.array(frame['observation.images.cam_1'])

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import cv2

def save_frames_as_mp4(dataset, camera_key="observation.images.cam_1", output_path="output.mp4", fps=30):
    """Simple function to save LeRobot dataset frames as MP4"""
    
    # Collect all frames first
    frames = []
    print("Collecting frames...")
    
    for i, data in enumerate(dataset):
        # Each data[camera_key] contains multiple RGB frames
        rgb_sequence = data[camera_key]
        
        # Loop through each RGB frame in the sequence
        for rgb_frame in rgb_sequence:
            frame = np.array(rgb_frame)
            
            # Convert to uint8 if needed
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            frames.append(frame)
        
        if i % 10 == 0:
            print(f"Processed {i} sequences, total frames: {len(frames)}")
    
    print(f"Total frames collected: {len(frames)}")
    
    if not frames:
        print("No frames found!")
        return
    
    # Get video dimensions from first frame
    first_frame = frames[0]
    if len(first_frame.shape) == 3:
        height, width, channels = first_frame.shape
    else:
        height, width = first_frame.shape
        channels = 1
    
    print(f"Video dimensions: {width}x{height}, {channels} channels")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write all frames
    print("Writing video...")
    for i, frame in enumerate(frames):
        # Convert RGB to BGR if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        video_writer.write(frame)
        
        if i % 100 == 0:
            print(f"Written {i}/{len(frames)} frames")
    
    video_writer.release()
    print(f"Video saved: {output_path}")

# Usage - just change your existing code to this:
dataset = LeRobotDataset("rdteteam/hello10_from_azure")

# Save as MP4
save_frames_as_mp4(dataset, "observation.images.cam_1", "hello10_cam1.mp4", fps=30)

# If you want to check frame types first:
# for data in dataset:
#     frame = np.array(data['observation.images.cam_1'])
#     print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
#     break  # Just check first frame