import pyrealsense2 as rs
import numpy as np
import torch
from ultralytics import YOLO
import serial
import time


# -------------------- UART FUNCTION --------------------

def uart_send_and_receive(uart, cmd):
    try:
        uart.write(bytes([cmd]))
        time.sleep(0.002)

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


# -------------------- YOLO INIT (JETSON) --------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("best.pt")
model.to(device)
print(f"YOLO running on: {device}")


# -------------------- MOTOR COMMANDS --------------------

MOTOR_STOP = 0x00
MOTOR_TURN_LEFT = 0x01
MOTOR_TURN_RIGHT = 0x02
MOTOR_GO_FRONT = 0x03


# -------------------- UART INIT (JETSON) --------------------

try:
    uart = serial.Serial(
        port='/dev/ttyTHS1',
        baudrate=115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0
    )
    time.sleep(2)
    uart.reset_input_buffer()
    uart.reset_output_buffer()
    uart_connected = True
    print("UART connected on /dev/ttyTHS1")
except Exception as e:
    print(f"UART connection failed: {e}")
    uart_connected = False


# -------------------- MAIN LOOP (NO UI) --------------------

last_print_time = 0.0

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        results = model(frame, conf=0.5, verbose=False)

        if results[0].boxes:
            box = results[0].boxes[0]
            xyxy = box.xyxy[0]
            center_x = float((xyxy[0] + xyxy[2]) / 2)
            frame_center_x = frame.shape[1] / 2

            if center_x < frame_center_x - 50:
                motor_command = MOTOR_TURN_RIGHT
                state = "TURN_RIGHT"
            elif center_x > frame_center_x + 50:
                motor_command = MOTOR_TURN_LEFT
                state = "TURN_LEFT"
            else:
                motor_command = MOTOR_GO_FRONT
                state = "GO_FRONT"
        else:
            motor_command = MOTOR_TURN_LEFT
            state = "SEARCH_LEFT"

        if uart_connected:
            resp = uart_send_and_receive(uart, motor_command)

            now = time.time()
            if now - last_print_time > 0.2:
                print(f"sent={motor_command} ({state}), recv={resp}")
                last_print_time = now

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    pipeline.stop()
    if uart_connected:
        uart.close()
    print("Cleanup complete")
