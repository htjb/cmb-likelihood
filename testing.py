
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
#noise /= np.sqrt(2*l+1)

fig, axes = plt.subplots(1, 2, figsize=(13,5), sharey=True)

# build the likelihood function with planck data
likelihood = cmb_generator.get_likelihood(planck, l, noise)
# evaluate planck likelihood at the planck best fit parameters

axes[0].plot(l, planck*(l*(l+1)/(2*np.pi)),label='Planck Data')
axes[0].plot(l, noise*(l*(l+1)/(2*np.pi)), label='Planck Noise Estimate (My Code)')
planck += noise

print(likelihood(theta))
print(likelihood(theta_bad))

"""realistic = np.loadtxt('planck_realistic_noise_from_montepython.txt')
rl = realistic[:,0]
nl = realistic[:,1]
nl *= (rl*(rl+1)/(2*np.pi))

axes[0].plot(rl, nl,label='Planck Realistic Noise (Monte Python)')"""

axes[0].grid()
axes[0].set_xlabel(r'$\ell$')
axes[0].set_ylabel(r'$\ell(\ell+1)C_\ell/(2\pi)$')
axes[0].legend()


cl, sample = cmb_generator.get_samples(l, theta, noise)

planck -= noise
cl -= noise
sample -= noise

axes[1].plot(l, planck*(l*(l+1)/(2*np.pi)),label='Planck Data')
axes[1].plot(l, cl*(l*(l+1)/(2*np.pi)), label='CAMB Model\n (Planck Best Fit Params)')
axes[1].plot(l, sample*(l*(l+1)/(2*np.pi)), label='Sampled Model')
axes[1].legend()
axes[1].set_title('With My Noise')
axes[1].loglog()
axes[0].loglog()


"""cl, sample = cmb_generator.get_samples(l, theta)

axes[2].plot(l, planck*(l*(l+1)/(2*np.pi)),label='Planck Data')
axes[2].plot(l, cl*(l*(l+1)/(2*np.pi)), label='CAMB Model\n (Planck Best Fit Params)')
axes[2].plot(l, sample*(l*(l+1)/(2*np.pi)), label='Sampled Model')
axes[2].legend()
axes[2].set_title('No Noise')"""

axes[1].set_xlabel(r'$\ell$')
#axes[2].set_xlabel(r'$\ell$')
axes[1].grid()
#axes[2].grid()

plt.tight_layout()
plt.savefig('planck_noise_comparison.png', dpi=300)
plt.legend()

plt.show()
