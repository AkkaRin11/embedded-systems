import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import serial
import time

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

model = YOLO("best.pt")
model.to("cuda" if torch.cuda.is_available() else "cpu")

# UART Configuration for STM32
# Motor control codes (1 byte)
MOTOR_STOP = 0x00
MOTOR_TURN_LEFT = 0x01
MOTOR_TURN_RIGHT = 0x02
MOTOR_GO_FRONT = 0x03

# Initialize UART (adjust COM port and baud rate as needed)
try:
    uart = serial.Serial(port='/dev/ttyTHS1', baudrate=115200, timeout=1)
    time.sleep(2)  # Wait for serial connection to stabilize
    uart_connected = True
except Exception as e:
    print(f"UART connection failed: {e}")
    uart_connected = False

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())

    results = model(frame, conf=0.5, verbose=False)
    annotated = results[0].plot()

    # Check for detected cups
    if results[0].boxes:
        # Get the first detected box
        box = results[0].boxes[0]
        xyxy = box.xyxy[0]  # Get coordinates [x1, y1, x2, y2]
        center_x = (xyxy[0] + xyxy[2]) / 2
        frame_center_x = frame.shape[1] / 2

        # Determine direction to turn
        if center_x < frame_center_x - 50:
            direction = "Turn Right"
            motor_command = MOTOR_TURN_RIGHT
            cv2.putText(annotated, ">", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        elif center_x > frame_center_x + 50:
            direction = "Turn Left"
            motor_command = MOTOR_TURN_LEFT
            cv2.putText(annotated, "<", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            direction = "Center"
            motor_command = MOTOR_GO_FRONT
            cv2.putText(annotated, "^", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        direction = "No Cups Detected"
        motor_command = MOTOR_TURN_LEFT
        cv2.putText(annotated, "<", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    # Send command to STM32 via UART
    if uart_connected:
        try:
            uart.write(bytes([motor_command]))
        except Exception as e:
            print(f"UART write error: {e}")

    cv2.imshow("YOLO + RS", annotated)

    if cv2.waitKey(1) == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()

# Close UART connection
if uart_connected:
    uart.close()
