import numpy as np
import camb
from scipy.stats import chi2
from pypolychord.priors import UniformPrior


class CMB():
    def __init__(self):

        """
        Class to generate CMB power spectra from CAMB, build and sample
        a likelihood function for a given data set and noise.
        """
        
        self.pars = camb.CAMBparams()
        
    def prior(self, cube):

        """
        Prior on the cosmological parameters 
        modified from https://arxiv.org/abs/1902.04029.

        Parameters
        ----------
        cube: array
            Array of values between 0 and 1.
        
        Returns
        -------
        theta: array
            Array of cosmological parameters.

        """

        theta = np.zeros(len(cube))
        theta[0] = UniformPrior(0.01, 0.085)(cube[0]) # omegabh2
        theta[1] = UniformPrior(0.08, 0.21)(cube[1]) # omegach2
        theta[2] = UniformPrior(0.97, 1.5)(cube[2]) # 100*thetaMC
        theta[3] = UniformPrior(0.01, 0.16)(cube[3]) # tau
        theta[4] = UniformPrior(0.8, 1.2)(cube[4]) # ns
        theta[5] = UniformPrior(2.6, 3.8)(cube[5]) # log(10^10*As)
        return theta
    
    def get_camb_model(self, theta):

        """
        Function to get the CAMB model for the CMB power spectrum.

        Parameters
        ----------
        theta: array
            The cosmological parameters as generated from self.prior.
        
        Returns
        -------
        cl: array
            The CMB power spectrum.
        """

        self.pars.set_cosmology(ombh2=theta[0], omch2=theta[1],
                                tau=theta[3], cosmomc_theta=theta[2]/100,
                                theta_H0_range=[5, 1000])
        self.pars.InitPower.set_params(As=np.exp(theta[5])/10**10, ns=theta[4])
        self.pars.set_for_lmax(2500, lens_potential_accuracy=0)

        results = camb.get_background(self.pars) # computes evolution of background cosmology

        cl = results.get_cmb_power_spectra(self.pars, CMB_unit='muK')['total'][:,0]
        cl *= (2*np.pi)/(np.arange(len(cl))*(np.arange(len(cl))+1)) # convert to C_l
        return cl
    
    def get_likelihood(self, data, l, noise=None):

        """
        Function to build a likelihood for a given data set and noise.

        Parameters
        ----------
        data: array
            The data set e.g. planck from `self.get_planck()`.
        
        l: array
            The multipoles associated with the data.
        
        noise: array
            The noise associated with the data. If None then no noise is added.
        
        Returns
        -------
        likelihood: function
            The likelihood function.

        """

        # if noise power add to the data 
        if noise is not None:
            data = data + noise

        def likelihood(theta):

            """
            The likelihood function. Generates a realisation of the
            CMB power spectrum from the CAMB model and compares it
            to data. Set up to be used with PolyChord.

            Parameters
            ----------
            theta: array
                The cosmological parameters as generated from self.prior.
            
            Returns
            -------
            logL: float
                The log likelihood.
            """

            cl = self.get_camb_model(theta)
            cl = np.interp(l, np.arange(len(cl)), cl)

            if noise is not None:
                cl += noise
            
            # likelihood is a reduced chi squared distribution
            x = (2*l + 1)* data/cl
            logL = -0.5*(-2*chi2(2*l+1).logpdf(x) 
                    - 2*np.log((2*l+1)/cl)).sum()
            
            return logL, []
        return likelihood
    
    def get_samples(self, l, theta, noise=None):

        """
        Code to generate observations of a theoretical CMB power spectrum.

        Parameters
        ----------
        l: array
            The multipoles.
        
        theta: array
            The cosmological parameters as generated from self.prior.
        
        noise: array
            The noise associated with the data. If None then no noise is added.
        """

        cl = self.get_camb_model(theta)
        cl = np.interp(l, np.arange(len(cl)), cl)

        # if noise then add noise
        if noise is not None:
            cl = cl + noise

        # draw a sample of (2l+1)*obs/theory from a chi2 distribution
        sample = chi2.rvs(df=2*l + 1, size=len(l))
        sample *= cl # multiply by theory
        sample /= (2*l + 1) # divide by 2l+1

        if noise is not None:
            # because the noise is well known subtract it from the sample
            cl = cl - noise # remove noise from theory
            sample = sample - noise # remove noise from sample
        
        return cl, sample