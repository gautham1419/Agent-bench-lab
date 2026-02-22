import pyRAPL
import time

try:
    pyRAPL.setup()

    meter = pyRAPL.Measurement('test')
    meter.begin()

    # simulate workload
    for _ in range(10_000_000):
        pass

    meter.end()

    print("Energy result:", meter.result)

except Exception as e:
    print("Error:", e)

