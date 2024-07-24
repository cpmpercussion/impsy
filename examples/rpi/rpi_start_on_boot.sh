#!/bin/sh

# This script installs systemd services to start IMPSY program on boot.
IMPSYROOT=/home/pi/imps/
IMPSYSERVICEDIR=examples/rpi
SYSTEMDDIR=/etc/systemd/system
IMPSYRUN=impsy-run.service
IMPSYWEB=impsy-web.service

cd ${IMPSYROOT}/${IMPSYSERVICEDIR}

# Enable IMPSY web UI server service
sudo cp ${IMPSYWEB} ${SYSTEMDDIR}
sudo chmod 644 ${SYSTEMDDIR}/${IMPSYWEB}
sudo systemctl enable ${IMPSYWEB}

# Enable IMPSY interaction server service
sudo cp ${IMPSYRUN} ${SYSTEMDDIR}
sudo chmod 644 ${SYSTEMDDIR}/${IMPSYRUN}
sudo systemctl enable ${IMPSYRUN}

# sudo systemctl start impsy-run.service
# sudo systemctl stop impsy-run.service

