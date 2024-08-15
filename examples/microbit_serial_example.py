from microbit import *
import utime
import math

def norm_acc(x):
    new = round(min(max((x + 2000)/4000, 0.0), 1.0), 4)
    return str(new)

last_acc_msg = ""

def send_accelerometer_data():
    global last_acc_msg
    if accelerometer.get_strength() > 1500:
        x, y, z = accelerometer.get_values()
        accs = map(norm_acc, [x,y,z])
        out = ','.join(accs)
        if out != last_acc_msg:
            print(out)
            last_acc_msg = out

def receive_and_display():
    if uart.any():
        data = uart.readline()
        if data is None:
            return
        data = data.decode('utf-8').strip().split(',')
        if len(data) == 3:
            try: 
                x, y, z = map(float, data)
                display.clear()
                display.set_pixel(0, 4 - min(math.floor((x + 0.2) * 4), 4), 9)
                display.set_pixel(2, 4 - min(math.floor((y + 0.2) * 4), 4), 9)
                display.set_pixel(4, 4 - min(math.floor((z + 0.2) * 4), 4), 9)
            except Exception as e:
                pass

uart.init(baudrate=115200)

while True:
    send_accelerometer_data()
    receive_and_display()
