#!/bin/bash
cd /home/pi/creative-mdns
# Start Pd
./start_pd.sh
# Start the RNN Box controller
python3 run_rnn_box.py -c
# After the RNN box controller exits, stop Pd
pkill -u pi pd
