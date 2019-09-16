

"""
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-6-120

When and why do we need data normalization?. Available from: 
https://www.researchgate.net/post/When_and_why_do_we_need_data_normalization 
[accessed Apr 12, 2017].

I use data [(x-mean)/sd] normalization whenever differences in variable ranges 
could potentially affect negatively to the performance of my algorithm. This is 
the case of PCA, regression or simple correlation analysis for example.

I use [x/max] when I'm just interested in some internal structure of the samples
and not in the absolute differences between samples. This might be the case of 
peak detection in spectra for samples in which the strength of the signal which 
I'm seeking changes from sample to sample.

Finally I use [x-mean] normalization when some samples could be potentially 
using just a part of a bigger scale. This is the case of ratings for movies for 
example, in which by some user tend to give more positive ratings than others. 

https://stats.stackexchange.com/questions/125259/normalize-variables-for-calculation-of-correlation-coefficient

Correlations of any kind automatically adjust for differing location and scale 
of variables, so any kind of linear scaling is unnecessary, but harmless. But if 
you are asking this then more study of standard text or internet sources to be 
clear on what correlations are seems indicated!
"""