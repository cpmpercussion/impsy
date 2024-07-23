#!/bin/sh

# This script installs a systemd service to start the genAI program on boot.

cd /home/pi/imps
sudo cp /home/pi/imps/examples/rpi/impsy.service /etc/systemd/system/impsy.service
sudo chmod 644 /etc/systemd/system/impsy.service
# sudo systemctl start impsy.service
# sudo systemctl stop impsy.service
sudo systemctl enable impsy.service
