from microbit import *
import utime
import math
import music

def norm_acc(x):
    new = round(min(max((x + 2000)/4000, 0.0), 1.0), 4)
    return new

last_acc_msg = ""
last_values = [0, 0, 0]
last_played_freqs = [0, 0, 0]

def send_accelerometer_data():
    global last_acc_msg
    global last_values
    if accelerometer.get_strength() > 1500:
        x, y, z = accelerometer.get_values()
        accs = [norm_acc(i) for i in [x,y,z]]
        last_values = accs
        accs = [str(i) for i in accs]
        out = ','.join(accs)
        if out != last_acc_msg:
            print(out)
            last_acc_msg = out

def receive_and_display():
    global last_values
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
                last_values = [x, y, z]
            except Exception as e:
                pass

def float_to_freq(value_in):
    """maps a float 0-1 to a frequency."""
    base = 220
    top = 880
    out = int(base + (value_in * (top - base)))
    out = max(0, min(out, 999))
    return out

def play_freqs(freqs):
    global last_played_freqs
    if freqs != last_played_freqs:
        eff = audio.SoundEffect(
            freq_start=freqs[0], 
            freq_end=freqs[1], 
            duration=freqs[2],
            vol_start=255,
            vol_end=255
        )
        audio.play(eff, wait=False)
        last_played_freqs = freqs
        # print("played: " + str(freqs))

def play_values():
    global last_played_freqs
    global last_values
    freqs = [0, 0, 0]
    if len(last_values) == 3:
        freqs = [float_to_freq(x) for x in last_values]
        play_freqs(freqs)



speaker.on()
# audio.play(Sound.MYSTERIOUS)
uart.init(baudrate=115200)

while True:
    send_accelerometer_data()
    receive_and_display()
    play_values()
