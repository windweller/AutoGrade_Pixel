"""
We use LSTM in two ways:
one as a generative model p(y1...yt|x) factorized
one as a discriminative model

We use Baselines in two ways:
one as
"""

from torch import nn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

assert pyro.__version__.startswith('1.4.0')



class MarkovBaseline(object):
    pass
