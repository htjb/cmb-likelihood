import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from cmblike.noise import wmap_noise, planck_noise
from cmblike.data import get_data
from cmblike.cmb import CMB

p, lplanck = get_data().get_planck()
noise, l = planck_noise().calculate_noise()
plt.plot(l, noise*l*(l+1)/(2*np.pi), label='Planck Noise')

cmbs = CMB(path_to_cp='/Users/harrybevins/Documents/Software/cosmopower')

theta = np.array([0.0224, 0.120, 0.054, 0.965, 3.0, 0.674])
parameters = ['omegabh2', 'omegach2', 'tau', 'ns', 'As', 'h']

lowl_bins = np.array([np.arange(2, 31), np.arange(2, 31)]).T
highl_bins = np.array([np.arange(30, 2500-30, 30), 
                       np.arange(60, 2500, 30)]).T
bins = np.vstack((lowl_bins, highl_bins))
l = np.array([(bins[i, 1] - bins[i, 0])/2 + bins[i, 0] 
              for i in range(len(bins))])

cl, sample = cmbs.get_samples(l, theta, noise=noise, cp=True, bins=bins)

plt.plot(l, sample*l*(l+1)/(2*np.pi), label='Sim. Observation')
plt.plot(lplanck, p*lplanck*(lplanck+1)/(2*np.pi), label='Data')
plt.legend()
#plt.loglog()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell(\ell+1)C_\ell/2\pi$')
plt.savefig('Example_planck.pdf', bbox_inches='tight')
plt.show()