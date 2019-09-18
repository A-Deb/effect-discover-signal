# effect-forecasting-models
This is a repository for sharing code and data with fellow researchers.

Models
======
This repository contains following models for predicting expected number of cyber attacks.


### total_eval.py
This file will perform the test on the external signals in the data folder.
Here you can set the 'event_type', 'organization', and 'method' selections. This file will run the eval_warning_replication.py script.

### eval_warning_replication.py
This file will perform the warning generation either at the freq='W' (weekly) or the freq='M' (monthly) level. It does so by generating the dates that will be used for the timeseries prediction.

### warning_replication.py
This file with either call ARIMA (for the baseline), ARIMAX (for ARIMA with external signal) or GRU (for GRU with external signal).

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


