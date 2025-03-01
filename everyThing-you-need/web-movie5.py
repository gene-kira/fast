import cv2
import numpy as np
import json
import logging
from queue import Queue, Empty
from threading import Thread, Event
from tqdm import tqdm
import torch
from moviepy.editor import VideoFileClip, AudioFileClip

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_deblurring_model(model_path):
    try:
        model = torch.load(model_path)
        model.eval()
        return model
    except Exception as e:
        logging.error(f"Error loading deblurring model: {e}")
        raise

def deblur_batch(frames, model, device):
    try:
        if not frames:
            return []
        
        # Convert frames to tensors
        input_tensors = []
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
            input_tensors.append(tensor)
        
        input_batch = torch.cat(input_tensors, dim=0)
        
        # Perform inference
        with torch.no_grad():
            output_batch = model(input_batch)
        
        # Convert tensors back to frames
        deblurred_frames = []
        for tensor in output_batch:
            frame_rgb = (tensor.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            deblurred_frames.append(frame_bgr)
        
        return deblurred_frames
    except Exception as e:
        logging.error(f"Error during deblur batch processing: {e}")
        raise

def apply_anomalies(frames, anomalies):
    try:
        enhanced_frames = []
        for frame in frames:
            if "noise" in anomalies:
                noise = np.random.normal(0, 25, frame.shape).astype(np.uint8)
                frame = cv2.add(frame, noise)
            if "grayscale" in anomalies:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            enhanced_frames.append(frame)
        return enhanced_frames
    except Exception as e:
        logging.error(f"Error during anomaly application: {e}")
        raise

def write_frames_to_video(frames, output_path, fps):
    try:
        if not frames:
            logging.error("No frames to write to video.")
            return
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
    except Exception as e:
        logging.error(f"Error writing frames to video: {e}")

def enhance_video(input_video_path, audio_file_path, output_video_path, anomalies, model, device, bitrate, resolution):
    try:
        # Load video
        video_clip = cv2.VideoCapture(input_video_path)
        if not video_clip.isOpened():
            logging.error("Error opening video file.")
            return
        
        fps = int(video_clip.get(cv2.CAP_PROP_FPS))
        width = int(video_clip.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_clip.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if resolution:
            width, height = resolution
            video_clip.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            video_clip.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        frames = []
        while True:
            ret, frame = video_clip.read()
            if not ret:
                break
            frames.append(frame)
        
        video_clip.release()
        
        # Create queues for frame processing and writing
        processing_queue = Queue(maxsize=50)
        writing_queue = Queue(maxsize=50)
        stop_event = Event()
        
        # Worker function for frame processing
        def process_frame_worker():
            while not stop_event.is_set():
                try:
                    frame_batch = processing_queue.get(timeout=1)
                    if frame_batch is None:
                        break
                    
                    if "deblur" in anomalies:
                        deblurred_frames = deblur_batch(frame_batch, model, device)
                    else:
                        deblurred_frames = frame_batch
                    
                    enhanced_frames = apply_anomalies(deblurred_frames, anomalies)
                    
                    for frame in enhanced_frames:
                        writing_queue.put(frame)
                except Empty:
                    continue
                except Exception as e:
                    logging.error(f"Error processing frame batch: {e}")
        
        # Worker function for frame writing
        def write_frame_worker():
            frames_to_write = []
            while not stop_event.is_set():
                try:
                    frame = writing_queue.get(timeout=1)
                    if frame is None:
                        break
                    
                    frames_to_write.append(frame)
                    
                    if len(frames_to_write) >= 50 or writing_queue.empty():
                        write_frames_to_video(frames_to_write, 'temp_video.mp4', fps)
                        frames_to_write.clear()
                except Empty:
                    continue
                except Exception as e:
                    logging.error(f"Error writing frame batch: {e}")
        
        # Start worker threads
        num_processing_threads = 4
        num_writing_threads = 2
        
        processing_threads = []
        for _ in range(num_processing_threads):
            thread = Thread(target=process_frame_worker, daemon=True)
            thread.start()
            processing_threads.append(thread)
        
        writing_threads = []
        for _ in range(num_writing_threads):
            thread = Thread(target=write_frame_worker, daemon=True)
            thread.start()
            writing_threads.append(thread)
        
        # Batch frames and put them into the processing queue
        batch_size = 16  # Adjust based on GPU memory capacity
        num_frames = len(frames)
        for i in tqdm(range(0, num_frames, batch_size), desc="Processing Frames"):
            frame_batch = frames[i:i + batch_size]
            processing_queue.put(frame_batch)
        
        # Wait for all frames to be processed
        for _ in range(num_processing_threads):
            processing_queue.put(None)
        
        for thread in processing_threads:
            thread.join()
        
        # Signal the writing threads to stop
        for _ in range(num_writing_threads):
            writing_queue.put(None)
        
        for thread in writing_threads:
            thread.join()
        
        # Add audio to the video using moviepy
        temp_clip = VideoFileClip('temp_video.mp4')
        audio_clip = AudioFileClip(audio_file_path)
        
        final_clip = temp_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_video_path, codec='libx264', bitrate=bitrate)
        
    except Exception as e:
        logging.error(f"Error enhancing video: {e}")
    finally:
        # Ensure all resources are released
        if 'temp_video.mp4' in locals():
            os.remove('temp_video.mp4')

def validate_config(config):
    required_keys = ['input_video', 'audio_file', 'output_video', 'model_path']
    for key in required_keys:
        if key not in config or not config[key]:
            logging.error(f"Missing or empty required configuration parameter: {key}")
            return False
    
    if 'resolution' in config and config['resolution']:
        try:
            width, height = map(int, config['resolution'].split('x'))
            if width <= 0 or height <= 0:
                raise ValueError
        except Exception as e:
            logging.error(f"Invalid resolution format: {config['resolution']}")
            return False
    
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Video Enhancement Script")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        if not validate_config(config):
            return
        
        input_video_path = config['input_video']
        audio_file_path = config['audio_file']
        output_video_path = config['output_video']
        model_path = config['model_path']
        anomalies = config.get('anomalies', [])
        bitrate = config.get('bitrate', '1000k')
        resolution = config.get('resolution', None)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_deblurring_model(model_path).to(device)
        
        enhance_video(input_video_path, audio_file_path, output_video_path, anomalies, model, device, bitrate, resolution)
    
    except Exception as e:
        logging.error(f"Error in main function: {e}")

if __name__ == "__main__":
    main()
