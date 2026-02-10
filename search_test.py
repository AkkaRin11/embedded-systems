import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import serial
import time

# -------------------- UART FUNCTION --------------------

def uart_send_and_receive(uart, cmd):
    try:
        uart.write(bytes([cmd]))
        time.sleep(0.002)  # allow STM32 time to echo

        if uart.in_waiting >= 1:
            resp = uart.read(1)
            return resp[0]
        return None

    except Exception as e:
        print(f"UART error: {e}")
        return None


# -------------------- REALSENSE INIT --------------------

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# -------------------- YOLO INIT --------------------

model = YOLO("best.pt")
model.to("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- MOTOR COMMANDS --------------------

MOTOR_STOP = 0x00
MOTOR_TURN_LEFT = 0x01
MOTOR_TURN_RIGHT = 0x02
MOTOR_GO_FRONT = 0x03

# -------------------- UART INIT --------------------

try:
    uart = serial.Serial(
        port='/dev/ttyTHS1',
        baudrate=115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0  # non-blocking
    )
    time.sleep(2)
    uart.reset_input_buffer()
    uart.reset_output_buffer()
    uart_connected = True
    print("UART connected")
except Exception as e:
    print(f"UART connection failed: {e}")
    uart_connected = False

# -------------------- MAIN LOOP --------------------

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())
    results = model(frame, conf=0.5, verbose=False)
    annotated = results[0].plot()

    # -------------------- OBJECT LOGIC --------------------

    if results[0].boxes:
        box = results[0].boxes[0]
        xyxy = box.xyxy[0]
        center_x = (xyxy[0] + xyxy[2]) / 2
        frame_center_x = frame.shape[1] / 2

        if center_x < frame_center_x - 50:
            motor_command = MOTOR_TURN_RIGHT
            cv2.putText(annotated, ">", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        elif center_x > frame_center_x + 50:
            motor_command = MOTOR_TURN_LEFT
            cv2.putText(annotated, "<", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            motor_command = MOTOR_GO_FRONT
            cv2.putText(annotated, "^", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        motor_command = MOTOR_TURN_LEFT
        cv2.putText(annotated, "<", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    # -------------------- UART SEND + RECEIVE --------------------

    if uart_connected:
        resp = uart_send_and_receive(uart, motor_command)

        if resp is not None:
            if resp == motor_command:
                status = "ACK"
                color = (0, 255, 0)
            else:
                status = f"WRONG:{resp}"
                color = (0, 0, 255)
        else:
            status = "NO RESP"
            color = (0, 0, 255)

        cv2.putText(annotated, status, (200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # -------------------- DISPLAY --------------------

    cv2.imshow("YOLO + RS + UART", annotated)

    if cv2.waitKey(1) == ord('q'):
        break

# -------------------- CLEANUP --------------------

pipeline.stop()
cv2.destroyAllWindows()

if uart_connected:
    uart.close()
