# Musical MDNs

Experiments with Mixture Density Networks for generating musical data.

![Musical MDN Example](https://github.com/cpmpercussion/musical-mdns/raw/master/images/mdn-output.png)

In this work musical data is considered to consist a time-series of continuous valued events. We seek to model the values of the events as well as the time in between each one. That means that these networks model data of at least two dimensions (event value and time).

Multiple implementations of a mixture density recurrent neural network are included for comparison.