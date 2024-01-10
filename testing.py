
import matplotlib.pyplot as plt
from tools.cmb import CMB
import numpy as np

# the generator
cmb_generator = CMB()

# planck best fit paraemters
theta = [0.022, 0.12, 1.04, 0.06, 0.96, 3.0]
theta_bad = [0.01146527, 0.09108639, 1.08655005, 
             0.1493919,  1.1031649,  3.30260607]
# get the planck data
planck, l, ps, ns = cmb_generator.get_planck()
# get an estimate of the planck instrument noise
noise = cmb_generator.get_planck_noise(l)
noise /= np.sqrt(2*l+1)

plt.plot(l, planck*(l*(l+1)/(2*np.pi)),label='Planck Data')
plt.plot(l, noise*(l*(l+1)/(2*np.pi)), label='Planck Noise Estimate (My Code)')

realistic = np.loadtxt('planck_realistic_noise_from_montepython.txt')
rl = realistic[:,0]
nl = realistic[:,1]
nl *= (rl*(rl+1)/(2*np.pi))
#nl /= np.sqrt(2*rl+1)
plt.plot(rl, nl,label='Planck Realistic Noise (Monte Python)')

plt.loglog()
plt.grid()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell(\ell+1)C_\ell/(2\pi)$')
plt.legend()
plt.tight_layout()
plt.savefig('planck_noise_comparison.png', dpi=300)
plt.legend()

plt.show()

# build the likelihood function with planck data
likelihood = cmb_generator.get_likelihood(planck, l, noise)
# evaluate planck likelihood at the planck best fit parameters
print(likelihood(theta))
print(likelihood(theta_bad))
