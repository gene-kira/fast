# real_time_ad_blocker.py

import cv2
import numpy as np
import librosa
import sounddevice as sd
import requests
import re
import threading
import time
from queue import Queue
from scipy.stats import entropy
from sklearn.externals import joblib

class FrameAnalyzer:
    def __init__(self, logo_templates, scene_model_path):
        self.logo_templates = [cv2.imread(p, 0) for p in logo_templates]
        self.scene_changes = []
        self.logo_detected = False
        self.scene_change_model = joblib.load(scene_model_path)

    def process_frame(self, frame, prev_gray=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            if np.mean(diff) > 50:
                self.scene_changes.append(True)
        for template in self.logo_templates:
            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            if np.max(res) > 0.8:
                self.logo_detected = True
        return gray

class AudioAnalyzer:
    def __init__(self, audio_patterns, model_path):
        self.audio_patterns = audio_patterns
        self.audio_matches = []
        self.audio_model = joblib.load(model_path)

    def capture_and_match(self, duration=3, fs=22050):
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        y = audio.flatten()
        for path in self.audio_patterns:
            y_pat, _ = librosa.load(path)
            corr = np.correlate(y, y_pat, mode='valid')
            if np.max(corr) > 0.7:
                self.audio_matches.append(path)

class DecisionEngine:
    def __init__(self, frame_analyzer, audio_analyzer):
        self.f = frame_analyzer
        self.a = audio_analyzer

    def detect_ads(self):
        if len(self.f.scene_changes) > 5 or self.f.logo_detected or self.a.audio_matches:
            return "Ad Detected"
        return "No Ad Detected"

    def context_aware_blocking(self):
        return "Disruptive Ad Blocked" if len(self.f.scene_changes) > 3 else "Non-Intrusive Ad Allowed"

    def detect_zero_day(self):
        return "Zero-Day Threat Detected" if self.context_aware_blocking() == "Disruptive Ad Blocked" else "Safe"

class SystemController:
    def __init__(self, video_src=0):
        self.cap = cv2.VideoCapture(video_src)
        self.prev_gray = None
        self.running = True
        self.audio_thread = None

        self.frame_analyzer = FrameAnalyzer(["brand_logo1.png", "brand_logo2.png"], "scene_change_model.pkl")
        self.audio_analyzer = AudioAnalyzer(["jingle1.wav", "jingle2.wav"], "audio_pattern_model.pkl")
        self.engine = DecisionEngine(self.frame_analyzer, self.audio_analyzer)

    def run(self):
        def audio_loop():
            while self.running:
                self.audio_analyzer.capture_and_match()
                time.sleep(5)  # Delay to avoid audio overload

        self.audio_thread = threading.Thread(target=audio_loop)
        self.audio_thread.start()

        while self.cap.isOpened() and self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            gray = self.frame_analyzer.process_frame(frame, self.prev_gray)
            self.prev_gray = gray

            # Decision after each frame
            result = self.engine.detect_ads()
            print(result)

            # Press 'q' to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        self.shutdown()

    def shutdown(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.running = False
        if self.audio_thread:
            self.audio_thread.join()

# Run it
if __name__ == "__main__":
    controller = SystemController(0)  # 0 for webcam or IP stream URL
    controller.run()

