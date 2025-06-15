
### Implementation

```python
import cv2
import numpy as np
import librosa
import time
from collections import deque

class VideoAdBlocker:
    def __init__(self, video_path, audio_path):
        self.video_path = video_path
        self.audio_path = audio_path
        self.logo_templates = ["brand_logo1.png", "brand_logo2.png"]
        self.audio_patterns = ["jingle1.wav", "jingle2.wav"]
        self.skip_button_template = cv2.imread("skip_button.png", 0)
        self.buffer_size = 300  # Number of frames to buffer
        self.frame_buffer = deque(maxlen=self.buffer_size)

    def detect_scene_changes(self):
        """Detects rapid scene transitions typical in ads."""
        cap = cv2.VideoCapture(self.video_path)
        prev_frame = None
        scene_changes = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frame_buffer.append((cap.get(cv2.CAP_PROP_POS_FRAMES), frame))
            
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray_frame)
                if np.mean(diff) > 50:  # Threshold for scene change
                    scene_changes.append(cap.get(cv2.CAP_PROP_POS_FRAMES))

            prev_frame = gray_frame

        cap.release()
        return scene_changes

    def detect_logo_overlays(self):
        """Identifies brand logos and promotional overlays."""
        cap = cv2.VideoCapture(self.video_path)
        logo_detected = False

        for template_path in self.logo_templates:
            template = cv2.imread(template_path, 0)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
                if np.max(res) > 0.8:  # Threshold for logo detection
                    logo_detected = True
                    break

        cap.release()
        return logo_detected

    def detect_audio_patterns(self):
        """Detects distinct jingles and sound cues."""
        y, sr = librosa.load(self.audio_path)
        detected_patterns = []

        for pattern_path in self.audio_patterns:
            y_pattern, sr_pattern = librosa.load(pattern_path)
            correlation = np.correlate(y, y_pattern, mode='valid')
            if np.max(correlation) > 0.7:  # Threshold for audio match
                detected_patterns.append(pattern_path)

        return detected_patterns

    def detect_skip_button(self):
        """Finds the 'skip' button in video ads."""
        cap = cv2.VideoCapture(self.video_path)
        skip_button_detected = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(gray_frame, self.skip_button_template, cv2.TM_CCOEFF_NORMED)
            if np.max(res) > 0.8:  # Threshold for skip button detection
                skip_button_detected = True
                break

        cap.release()
        return skip_button_detected

    def detect_ads(self):
        """Combines all detection methods to flag ads."""
        scene_changes = self.detect_scene_changes()
        logo_detected = self.detect_logo_overlays()
        audio_matches = self.detect_audio_patterns()
        skip_button_detected = self.detect_skip_button()

        if len(scene_changes) > 5 or logo_detected or len(audio_matches) > 0 or skip_button_detected:
            return "Ad Detected"
        return "No Ad Detected"

    def edit_video(self):
        """Edits out detected ads on the fly."""
        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Check for ad detection
            if self.detect_ads() == "Ad Detected":
                # Find the start of the ad by looking backwards in the buffer
                start_frame = None
                for i in range(len(self.frame_buffer) - 1, -1, -1):
                    if self.detect_ads():
                        start_frame = self.frame_buffer[i][0]
                        break

                # Skip the detected ad segment
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + len(scene_changes))

            else:
                out.write(frame)

        cap.release()
        out.release()

# Example Usage
video_ad_blocker = VideoAdBlocker("sample_video.mp4", "sample_audio.wav")
video_ad_blocker.edit_video()
```

### Explanation

1. **Scene Change Detection**: Detects rapid scene changes by comparing the difference between consecutive frames.
2. **Logo & Overlay Recognition**: Uses template matching to identify brand logos and promotional overlays.
3. **Audio Pattern Matching**: Loads audio patterns and correlates them with the video's