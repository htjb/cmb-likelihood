# Mock CMB Likelihood


## Introduction

| cmb_likelihood | Toy CMB Likelihood Code |
|----------------|-------------------------|
| Authors | Harry T.J. Bevins |
| Version | 0.1.0.beta |
| Homepage | |
|Documentation | |

## Installation

Currently can install the package with

```
git clone https://github.com/htjb/cmb-likelihood
pip install .
```

## The point

Given a C_l^{TT} data set and a noise power spectrum

```python
from cmblike.noise import planck_noise
from cmblike.data import get_data

planck, l = get_data().get_planck()
noise = planck_noise(l).calculate_noise()

```

the package allows users to;

- generate realisations of a CMB TT power spectrum 
    as if it had been observed by an 
    instrument with a user specified noise profile.
- generate theoretical models of the power spectrum
    with CAMB
- evaluate the likelihood for a theoretical model
    given a data set


## Citation