import cv2
import numpy as np

class ProcessTracker:
    def __init__(self):
        self.model = None  # For any machine learning model if needed in future

    def extract_features(self, process_stats):
        """
        Extracts relevant features from each process's start and end times.
        
        Args:
            process_stats (list): List of dictionaries containing 'start_time' and 'end_time'.
            
        Returns:
            numpy array: An array of durations in seconds for each process.
        """
        durations = []
        for proc in process_stats:
            start = proc['start_time'].timestamp()
            end = proc['end_time'].timestamp()
            duration = end - start
            durations.append(duration)
        
        return np.array(durations, dtype=np.float64)

    def track_processes(self, video_path):
        """
        Tracks processes using optimized computer vision techniques.
        
        Args:
            video_path (str): Path to the video file to process.
            
        Returns:
            bool: True if tracking completed successfully, False otherwise.
        """
        # Initialize Video Capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video stream or file")
            return False

        # Read first frame and extract initial window
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame")
            cap.release()
            return False
        
        h, w = 30, 50  # Example dimensions for the tracking window
        x, y = 280, 470  # Initial position of the tracking window
        track_window = (x, y, w, h)
        
        # Convert initial frame to HSV and create histogram
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros_like(hsv_frame)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
        # Calculate Histogram
        hist = cv2.calcHist([hsv_frame], [0, 1], mask, [90, 180], [0, 256, 0, 256])
        hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Convert to HSV and compute back projection
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = np.zeros_like(hsv_frame)
            
            # Back project using histogram
            cv2.calcBackProject([hsv_frame], [0, 1], hist, [90, 180], dst)
            
            # Apply Gaussian blur to smooth the result
            dst = cv2.GaussianBlur(dst, (15, 15), 0)
            
            # Use meanShift to track the window
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            rect, track_window = cv2.meanShift(dst, track_window, criteria)
            
            if track_window is not None:
                x, y, w, h = track_window
                # Draw tracking window on the original frame
                cv2.rectangle(frame, (int(x), int(y)), 
                            (int(x + w), int(y + h)), 255, 2)
            
            cv2.imshow('Tracking', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        return True

if __name__ == "__main__":
    tracker = ProcessTracker()
    video_path = "path_to_your_video_file.mp4"
    success = tracker.track_processes(video_path)
    print(f"Tracking completed: {success}")
