import numpy as np
import camb
from scipy.stats import chi2
from pypolychord.priors import UniformPrior


class CMB():
    def __init__(self):

        """
        """
        
        self.pars = camb.CAMBparams()
        
    def prior(self, cube):
        theta = np.zeros(len(cube))
        theta[0] = UniformPrior(0.01, 0.085)(cube[0]) # omegabh2
        theta[1] = UniformPrior(0.08, 0.21)(cube[1]) # omegach2
        theta[2] = UniformPrior(0.97, 1.5)(cube[2]) # 100*thetaMC
        theta[3] = UniformPrior(0.01, 0.16)(cube[3]) # tau
        theta[4] = UniformPrior(0.8, 1.2)(cube[4]) # ns
        theta[5] = UniformPrior(2.6, 3.8)(cube[5]) # log(10^10*As)
        return theta
    
    def get_planck(self):

        """
        Function to load in the planck power spectrum data.

        Returns
        -------
        p: power spectrum
        ps: the error on the power spectrum
        l_real: the multipoles
        """

        tt = np.loadtxt('TT_power_spec.txt', delimiter=',', dtype=str)

        l, p, ps, ns = [], [], [], []
        for i in range(len(tt)):
            if tt[i][0] == 'Planck binned      ':
                l.append(tt[i][2].astype('float')) # ell
                ps.append(tt[i][6].astype('float')) # error on power spectrum
                ns.append(tt[i][5].astype('float')) # power spectrum
                p.append(tt[i][4].astype('float')) # power spectrum
        p, l = np.array(p), np.array(l)
        ps, ns = np.array(ps), np.array(ns)
        p *= (2*np.pi)/(l*(l+1)) # convert to C_l
        return p, l, ps, ns
    
    def get_planck_noise(self, l):

        # from table 4 in https://arxiv.org/pdf/1807.06205.pdf
        theta_planck = np.array([9.66, 7.22, 4.90]) # beam size in arcmin
        theta_planck *= np.array([np.pi/60/180]) # convert to radians
        sigma_T = np.array([1.29, 0.55, 0.78])*np.pi/180 
        
        nis = []
        for i in range(len(sigma_T)):
            # from montepython code https://github.com/brinckmann/montepython_public/blob/3.6/montepython/likelihood_class.py#L1096
            ninst = sigma_T[i]**-2 * \
                np.exp(-l*(l+1)*theta_planck[i]**2/(8*np.log(2))) #one over ninst
            nis.append(ninst)
        ninst = np.array(nis).T
        ninst = np.sum(ninst, axis=1)
        noise = 1/ninst
        return noise#/np.sqrt(2*l+1)
    
    def get_camb_model(self, theta):
        self.pars.set_cosmology(ombh2=theta[0], omch2=theta[1],
                                tau=theta[3], cosmomc_theta=theta[2]/100,
                                theta_H0_range=[5, 1000])
        self.pars.InitPower.set_params(As=np.exp(theta[5])/10**10, ns=theta[4])
        self.pars.set_for_lmax(2500, lens_potential_accuracy=0)

        results = camb.get_background(self.pars) # computes evolution of background cosmology

        cl = results.get_cmb_power_spectra(self.pars, CMB_unit='muK')['total'][:,0]
        cl *= (2*np.pi)/(np.arange(len(cl))*(np.arange(len(cl))+1)) # convert to C_l
        return cl
    
    def get_likelihood(self, p, l, noise=None):
        def likelihood(theta):

            cl = self.get_camb_model(theta)
            cl = np.interp(l, np.arange(len(cl)), cl)

            if noise is not None:
                cl += noise
            
            x = (2*l + 1)* p/cl
            L = -0.5*(-2*chi2(2*l+1).logpdf(x) 
                    - 2*np.log((2*l+1)/cl)).sum()
            
            return L, []
        return likelihood
    
    def get_samples(self, l, theta, noise=None):

        cl = self.get_camb_model(theta)
        cl = np.interp(l, np.arange(len(cl)), cl)

        # if noise then add noise
        if noise is not None:
            cl += noise

        sample = chi2.rvs(df=2*l + 1, size=len(l))
        sample *= cl
        sample /= (2*l + 1)
        
        return cl, sample