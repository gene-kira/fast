# probe_types.py

import random

class BaseProbe:
    def __init__(self, pid, data_type="unknown"):
        self.pid = pid
        self.data_type = data_type
        self.signal = random.uniform(0.4, 1.0)
        self.status = "ready"

    def decay_signal(self):
        self.signal *= random.uniform(0.85, 0.95)
        return round(self.signal, 2)

    def retry(self):
        if self.signal < 0.6:
            self.signal *= 1.1

    def display_battery(self):
        level = int(self.signal * 5)
        return f"{'🔋'*level}{'▫️'*(5 - level)}"

class PulseProbe(BaseProbe):
    def __init__(self, pid):
        super().__init__(pid, "pulse.glyph")

class DeepProbe(BaseProbe):
    def __init__(self, pid):
        super().__init__(pid, "deep.glyph")

class ReflectiveProbe(BaseProbe):
    def __init__(self, pid):
        super().__init__(pid, "reflect.glyph")

