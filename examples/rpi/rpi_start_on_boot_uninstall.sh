#!/bin/sh

# This script uninstalls systemd services that start IMPSY program on boot.
SYSTEMDDIR=/etc/systemd/system
IMPSYRUN=impsy-run.service
IMPSYWEB=impsy-web.service

# Enable IMPSY web UI server service
sudo systemctl disable ${IMPSYWEB}
sudo systemctl stop ${IMPSYWEB}
sudo rm ${SYSTEMDDIR}/${IMPSYWEB}

# Enable IMPSY interaction server service
sudo systemctl disable ${IMPSYRUN}
sudo systemctl stop ${IMPSYRUN}
sudo rm ${SYSTEMDDIR}/${IMPSYRUN}
