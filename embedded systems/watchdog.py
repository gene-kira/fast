# watchdog.py
import time

class MagicBoxWatchdog:
    def __init__(self, dma, timeout=5):
        self.dma = dma
        self.timeout = timeout
        self.active = True

    def refresh(self):
        self.dma.last_trigger = time.time()

    def run(self):
        while self.active:
            if self.dma.dma_busy and (time.time() - self.dma.last_trigger > self.timeout):
                print("⚠️ DMA stalled. Watchdog reset issued.")
                self.dma.reset_dma()
            time.sleep(1)

