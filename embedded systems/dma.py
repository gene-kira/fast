# dma.py
import time

class MagicBoxDMA:
    def __init__(self, interface):
        self.interface = interface
        self.dma_busy = False
        self.last_trigger = time.time()

    def write_mem(self, address, data):
        cmd = f"WRITE {hex(address)} {data}\n"
        self.interface.send(cmd.encode())

    def read_mem(self, address):
        cmd = f"READ {hex(address)}\n"
        self.interface.send(cmd.encode())
        return self.interface.receive()

    def trigger_dma(self, src, dest, length):
        self.dma_busy = True
        self.last_trigger = time.time()
        cmd = f"DMA {hex(src)} {hex(dest)} {length}\n"
        self.interface.send(cmd.encode())
        time.sleep(length * 0.001)
        self.dma_busy = False
        return "DMA complete"

    def reset_dma(self):
        self.interface.send(b"RESET_DMA\n")

