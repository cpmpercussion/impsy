#!/bin/sh

## This script sets up the raspberry pi to:
# - support ethernet over USB with the interface usb0
# - support UART over GPIO for MIDI in/out

# Setup Ethernet gadget according to: https://forums.raspberrypi.com/viewtopic.php?p=2184846
cat >/etc/network/interfaces.d/g_ether <<'EOF'
auto usb0
allow-hotplug usb0
iface usb0 inet static
        address 169.254.1.107
        netmask 255.255.0.0

auto usb0.1
allow-hotplug usb0.1
iface usb0.1 inet dhcp

EOF

## Ethernet gadget setup:

# probably add to /boot/config.txt 
# dtoverlay=dwc2
sudo echo -e "\ndtoverlay=dwc2" >> /boot/firmware/config.txt

## TODO
# add to end of  /boot/firmware/cmdline.txt
sudo sed -i 's/$/ modules-load=dwc2,g_ether/' /boot/firmware/cmdline.txt
# modules-load=dwc2,g_ether

## Enable UART0
# https://www.raspberrypi.com/documentation/computers/configuration.html#uarts-and-device-tree
# Add this to /boot/firmware/config.txt: dtoverlay=disable-bt
sudo echo -e "\ndtoverlay=disable-bt" >> /boot/firmware/config.txt
sudo systemctl disable hciuart

# Reboot!
# sudo reboot
