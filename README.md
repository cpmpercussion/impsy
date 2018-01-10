# Musical MDNs

Experiments with Mixture Density Networks for generating musical data.

![Musical MDN Example](https://github.com/cpmpercussion/musical-mdns/raw/master/images/mdn-output.png)

In this work musical data is considered to consist a time-series of continuous valued events. We seek to model the values of the events as well as the time in between each one. That means that these networks model data of at least two dimensions (event value and time).

Multiple implementations of a mixture density recurrent neural network are included for comparison.

## Installing on Raspberry Pi

Some of these files are intended to be used on a Raspberry Pi. Installing Tensorflow on RPi is tricky given that there is no official build, however, various unofficial builds can be found.

For our work, we use `DeftWork`'s build of Tensorflow 1.3.0 as follows:

    wget https://github.com/DeftWork/rpi-tensorflow/raw/master/tensorflow-1.3.0-cp34-cp34m-linux_armv7l.whl
    sudo pip3 install tensorflow-1.3.0-cp34-cp34m-linux_armv7l.whl

There's build instructions for Raspberry courtesy of [samjabrahams](https://github.com/samjabrahams/tensorflow-on-raspberry-pi) and more info on a possible Docker solution from [DeftWork](https://github.com/DeftWork/rpi-tensorflow). Thanks internets!

In addition to tensorflow, you also need `pandas`, `numpy` and `pySerial`. Then the interface controller can be run like so:

    python3 musical_mdn_interface_controller.py
