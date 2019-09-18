# effect-forecasting-models
This is a repository for sharing code and data with fellow researchers.

Models
======
This repository contains following models for predicting expected number of cyber attacks.


### total_eval.py


### Baserate Model

Baserate model is simply a rolling mean of past cyber attacks. The estimated
mean can also be tought of the rate of a Poisson process. This rate is used for
out-of-sample forecasting. 
 
 
### ARIMA

ARIMA is an autoregressive model and is the acronym for Autoregressive
Integrated Moving Average. The key assumption for this model is that the
observation at a particular time point depends on immediate past observations
and past forecast errors. ARIMA is defined with three order terms: a) number of
autoregressive lags/order ($p$), b) number of difference operations used to
make the series stationary ($d$), and c) number of moving average terms ($q$).
These orders (p, d, q) are selected using a search over a grid (defined with
$max_p$, $max_d$, $max_q$) with min AIC score. For day-specific ARIMA, we
create an ARIMA model for the time series specific to a day and use the model
for predicting for the same day in future.


Dependencies
============
The models are developed and test with Python 3.6 and requires following
packages: 
* pandas
* scipy
* numpy
* sklearn
* matplotlib
* json
* elasticsearch
* argparse

Installation
============

1. Checkout the latest release

   You can see the latest releases at https://github.com/A-Deb/effect-discover-signal



Testing
=======

### Basrate

* Generate warnings with default settings.

```
python3 gen_warning_baserate.py
```


* Generate warnings for dexter malicious email with look-ahead of 7 days.

```
python3 gen_warning_baserate.py -d dexter_malicious-email -l 7 --sensor antivirus
```



### ARIMA

* To generate warnings with a forecast look-ahead of 7 days starting from 2017-06-15

```
python3 gen_warning_arima.py -d dexter_malicious-email -l 7 --event-type malicious-email --warn-start-date 2017-06-05
```


