import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import queue

# Ensure necessary libraries are installed
try:
    import numpy
    import cv2
    import torch
    from torchvision import transforms
except ImportError:
    print("Installing necessary libraries...")
    import subprocess
    subprocess.run(['pip', 'install', 'numpy', 'opencv-python', 'torch', 'torchvision'])

# Define the U-Net model for object enhancement
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        
        def conv_block(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )
        
        def up_conv(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU()
            )

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dec1 = up_conv(512, 256)
        self.dec2 = up_conv(256, 128)
        self.dec3 = up_conv(128, 64)
        
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        dec1 = self.dec1(enc4) + enc3
        dec2 = self.dec2(dec1) + enc2
        dec3 = self.dec3(dec2) + enc1

        out = self.out(dec3)
        return torch.tanh(out)

# Initialize the model and move it to GPU
model = UNet().to('cuda')
model.eval()

class FrameGenerator:
    def __init__(self, video_path, batch_size=16):
        self.video_path = video_path
        self.batch_size = batch_size
        self.cap = cv2.VideoCapture(video_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.frame_queue = queue.Queue(maxsize=100)  # Queue to hold frames
        self.result_queue = queue.Queue(maxsize=100)  # Queue to hold processed results

    def read_frames(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame).to(self.device)
            self.frame_queue.put(frame)

    def process_frames(self, batch_size):
        frames = []
        while True:
            try:
                frame = self.frame_queue.get_nowait()
                frames.append(frame)
                if len(frames) == batch_size:
                    break
            except queue.Empty:
                break

        if frames:
            batch = torch.stack(frames).to(self.device)
            with torch.no_grad():
                enhanced_frames = self.model(batch)
            for frame in enhanced_frames:
                frame = (frame * 0.5 + 0.5) * 255
                frame = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                self.result_queue.put(frame)

    def run(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            futures.append(executor.submit(self.read_frames))
            for _ in range(3):  # Process frames in parallel
                futures.append(executor.submit(self.process_frames, self.batch_size))

            while True:
                try:
                    frame = self.result_queue.get_nowait()
                    yield frame
                except queue.Empty:
                    if all(future.done() for future in futures):
                        break

if __name__ == "__main__":
    video_path = 'path_to_your_video.mp4'
    generator = FrameGenerator(video_path, batch_size=16)

    for frame in generator.run():
        # Display the enhanced frame using OpenCV
        cv2.imshow('Enhanced Frame', frame)
        
    cv2.destroyAllWindows()
