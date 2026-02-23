import time
import threading
import psutil
from pynvml import *
import pyRAPL


class ResourceMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.running = False

        self.cpu = []
        self.ram = []
        self.gpu_util = []
        self.gpu_mem = []
        self.gpu_power = []

        self.gpu_energy_joules = 0
        self.cpu_energy_joules = 0

    def start(self):
        # ---------- GPU INIT ----------
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(0)

        # ---------- CPU PROCESS ----------
        self.process = psutil.Process()

        # ---------- CPU ENERGY (RAPL) ----------
        pyRAPL.setup()
        self.cpu_meter = pyRAPL.Measurement('benchmark')
        self.cpu_meter.begin()

        # ---------- START THREAD ----------
        self.running = True
        self.thread = threading.Thread(target=self._collect)
        self.thread.start()

    def _collect(self):
        while self.running:
            # ----- CPU Util -----
            self.cpu.append(psutil.cpu_percent())

            # ----- RAM -----
            self.ram.append(self.process.memory_info().rss / 1024**2)

            # ----- GPU Util -----
            util = nvmlDeviceGetUtilizationRates(self.handle)
            self.gpu_util.append(util.gpu)

            # ----- GPU Memory -----
            mem = nvmlDeviceGetMemoryInfo(self.handle)
            self.gpu_mem.append(mem.used / 1024**2)

            # ----- GPU Power (Watts) -----
            power = nvmlDeviceGetPowerUsage(self.handle) / 1000  # mW → W
            self.gpu_power.append(power)

            # ----- GPU Energy Accumulation -----
            self.gpu_energy_joules += power * self.interval

            time.sleep(self.interval)

    def stop(self):
        # stop sampling
        self.running = False
        self.thread.join()

        # stop CPU energy measurement safely
        try:
            self.cpu_meter.end()

            if (
                self.cpu_meter.result is not None and
                self.cpu_meter.result.pkg is not None and
                len(self.cpu_meter.result.pkg) > 0
            ):
                self.cpu_energy_joules = (
                    self.cpu_meter.result.pkg[0] / 1_000_000
                )
            else:
                self.cpu_energy_joules = 0

        except Exception as e:
            print(f"[WARNING] CPU energy measurement failed: {e}")
            self.cpu_energy_joules = 0

        nvmlShutdown()

    def summary(self):
        return {
            "cpu_avg": sum(self.cpu)/len(self.cpu),
            "cpu_peak": max(self.cpu),

            "ram_avg": sum(self.ram)/len(self.ram),
            "ram_peak": max(self.ram),

            "gpu_util_avg": sum(self.gpu_util)/len(self.gpu_util),
            "gpu_util_peak": max(self.gpu_util),

            "gpu_mem_avg": sum(self.gpu_mem)/len(self.gpu_mem),
            "gpu_mem_peak": max(self.gpu_mem),

            "gpu_power_avg": sum(self.gpu_power)/len(self.gpu_power),

            # NEW ENERGY METRICS
            "gpu_energy_joules": self.gpu_energy_joules,
            "cpu_energy_joules": self.cpu_energy_joules,

            # Optional total energy
            "total_energy_joules": self.gpu_energy_joules + self.cpu_energy_joules
        }