import serial
import time

PORT = "/dev/ttyTHS1"
BAUD = 115200
TEST_BYTE = 0x03

def main():
    try:
        uart = serial.Serial(
            port=PORT,
            baudrate=BAUD,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0  # non-blocking
        )
        time.sleep(2)
        uart.reset_input_buffer()
        uart.reset_output_buffer()
        print(f"Opened {PORT} @ {BAUD}")
    except Exception as e:
        print(f"Failed to open UART: {e}")
        return

    try:
        while True:
            print("Sending 0x03...")
            uart.write(bytes([TEST_BYTE]))

            # Wait up to 1 second for echo
            start = time.time()
            received = None

            while time.time() - start < 1.0:
                if uart.in_waiting >= 1:
                    received = uart.read(1)[0]
                    break

            if received is None:
                print("No response")
            elif received == TEST_BYTE:
                print("ACK received: 0x03 ✅")
            else:
                print(f"Wrong byte received: 0x{received:02X} ❌")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nExiting test")

    finally:
        uart.close()

if __name__ == "__main__":
    main()
