import fusion
import time


# Example of how to use the fusion module
while True:
    now = int(time.time()*1000) - 1000
    roll, pitch, yaw = fusion.getEulerAngle(now)
    print(f"Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")
    time.sleep(0.1)