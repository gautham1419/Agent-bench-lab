import time
import threading
import psutil
from pynvml import *
import pyRAPL


class ResourceMonitor:
    def __init__(self, interval=0.5, cpu_reset_interval=300):
        self.interval = interval
        self.cpu_reset_interval = cpu_reset_interval  # seconds
        self.running = False

        self.cpu = []
        self.ram = []
        self.gpu_util = []
        self.gpu_mem = []
        self.gpu_power = []

        self.gpu_energy_joules = 0
        self.cpu_energy_joules = 0

        self.last_cpu_reset = None

    def start(self):
        # ---------- GPU INIT ----------
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(0)

        # ---------- CPU PROCESS ----------
        self.process = psutil.Process()

        # ---------- CPU ENERGY INIT ----------
        pyRAPL.setup()
        self._start_cpu_meter()

        # ---------- START THREAD ----------
        self.running = True
        self.thread = threading.Thread(target=self._collect)
        self.thread.start()

    def _start_cpu_meter(self):
        self.cpu_meter = pyRAPL.Measurement('benchmark')
        self.cpu_meter.begin()
        self.last_cpu_reset = time.time()

    def _accumulate_cpu_energy(self):
        try:
            self.cpu_meter.end()

            if (
                self.cpu_meter.result is not None and
                self.cpu_meter.result.pkg is not None and
                len(self.cpu_meter.result.pkg) > 0
            ):
                delta = self.cpu_meter.result.pkg[0] / 1_000_000
                self.cpu_energy_joules += delta

        except Exception as e:
            print(f"[WARNING] CPU chunk measurement failed: {e}")

    def _collect(self):
        while self.running:
            current_time = time.time()

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
            power = nvmlDeviceGetPowerUsage(self.handle) / 1000
            self.gpu_power.append(power)

            # ----- GPU Energy Accumulation -----
            self.gpu_energy_joules += power * self.interval

            # ----- CPU ENERGY CHUNK RESET -----
            if current_time - self.last_cpu_reset >= self.cpu_reset_interval:
                self._accumulate_cpu_energy()
                self._start_cpu_meter()

            time.sleep(self.interval)

    def stop(self):
        # stop sampling
        self.running = False
        self.thread.join()

        # accumulate final CPU chunk
        self._accumulate_cpu_energy()

        nvmlShutdown()

    def summary(self):
        return {
            "cpu_avg": sum(self.cpu)/len(self.cpu) if self.cpu else 0,
            "cpu_peak": max(self.cpu) if self.cpu else 0,

            "ram_avg": sum(self.ram)/len(self.ram) if self.ram else 0,
            "ram_peak": max(self.ram) if self.ram else 0,

            "gpu_util_avg": sum(self.gpu_util)/len(self.gpu_util) if self.gpu_util else 0,
            "gpu_util_peak": max(self.gpu_util) if self.gpu_util else 0,

            "gpu_mem_avg": sum(self.gpu_mem)/len(self.gpu_mem) if self.gpu_mem else 0,
            "gpu_mem_peak": max(self.gpu_mem) if self.gpu_mem else 0,

            "gpu_power_avg": sum(self.gpu_power)/len(self.gpu_power) if self.gpu_power else 0,

            "gpu_energy_joules": self.gpu_energy_joules,
            "cpu_energy_joules": self.cpu_energy_joules,
            "total_energy_joules": self.gpu_energy_joules + self.cpu_energy_joules
        }