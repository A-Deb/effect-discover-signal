# effect-discover-signals
This is a repository for sharing codes with IARPA.

Models
======
This repository contains following models for predicting expected number of cyber attacks.

### Baserate Model

Baserate model is simply a rolling mean of past cyber attacks. The estimated
mean can also be tought of the rate of a Poisson process. This rate is used for
out-of-sample forecasting. 

### Day Specific Baserate Model

Some types of cyber attacks show variation with weekdays. In addition to
baserate model, we propose day-specific baserate model. The key idea is to
estimate mean for each weekday. When forecasting attacks for a date, the model
uses correspoding rate for the day of that date.  
 

### Hidden Markov Model

We propose a hidden Markov model for forecasting the expected number of cyber
attacks, given the past time series of attacks. The number of hidden states is
parameterized. For a two-state HMM the states represent high and low activity
of cyber attacks, whereas for a three-state HM the states represent high, mid,
and low activity of cyber attacks. For modeling the count of attacks within a
state, we adopt any of the four distributions, which are Poisson, Gaussian,
geometric, and hurdle geometric. We use EM algorithm for learning the
parameters of the model. 

HMM model fetches data using Elastic search. Currently any of the two ransomware data,
locky and cerber, is fetched given the input option. Data is then processed to create
a time series for the given window. An HMM model is learned with user-defined hidden 
states and emission distribution, and the learned parameters are used to predict the 
expected number of events the next day.


### ARIMA and Day-specific ARIMA

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

   You can see the latest releases at https://github.com/usc-isi-i2/effect-forecasting-models

2. Build necessary dependencies for HMM

   From effect-forecasting-models root directory:
   ```
    ./install.sh
   ```

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

### Daywise Basrate

* Generate warnings with default settings.

```
python3 gen_warning_daywise_baserate.py
```


* Generate warnings for dexter malicious email with look-ahead of 7 days.

```
python3 gen_warning_daywise_baserate.py -d dexter_malicious-email -l 7 --sensor antivirus
```

### HMM

* Warning can be generated with default settings for HMM.

```
sh gen_warning_hmm.sh
```
or
```
python3 gen_warning_hmm.py
```

* To see the parameters of the model, execute the following commands:

```
python3 gen_warning_hmm.py -h
```
We can vary various parameters of the model. Some examples are as follows. 

* To change the data sourc use ```-d``` option. apply

```
python3 gen_warning_hmm.py -d ransomware_cerber
```

* To generate warnings for a HMM with three states and hurdle geometric emission distribution

```
python3 gen_warning_hmm.py -d ransomware_locky -z 3 -e HurdleGeometric
```

* To generate warnings with a forecast look-ahead of 7 days 

```
python3 gen_warning_hmm.py -d dexter_malicious-email -l 7 --event-type malicious-email --sensor antivirus
```

### ARIMA

* To generate warnings with a forecast look-ahead of 7 days starting from 2017-06-15

```
python3 gen_warning_arima.py -d dexter_malicious-email -l 7 --event-type malicious-email --warn-start-date 2017-06-05
```

### Day-specific ARIMA

* To generate warnings with a forecast look-ahead of 7 days starting from 2017-06-15

```
python3 gen_warning_daywise_arima.py -d dexter_malicious-url -l 7 --sensor antivirus --warn-start-date 2017-06-05 --event-type malicious-url
```

