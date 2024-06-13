import numpy as np
import camb
from scipy.stats import chi2
import cosmopower as cp
from pypolychord.priors import UniformPrior
import warnings

class CMB():
    def __init__(self, **kwargs):

        """
        Class to generate CMB power spectra from CAMB, build and sample
        a likelihood function for a given data set and noise.

        Parameters:
        -----------
        parameters: list
            The cosmological parameters to sample over. Defaults to
            ['omegabh2', 'omegach2', 'thetaMC', 'tau', 'ns', 'As'].
        prior_mins: list
            The minimum values for the priors. Defaults to
            [0.01, 0.08, 0.97, 0.01, 0.8, 2.6].
        prior_maxs: list
            The maximum values for the priors. Defaults to
            [0.085, 0.21, 1.5, 0.16, 1.2, 3.8].
        default_parameter_values: list
            The default values for the cosmological parameters (inspired by
            Planck observations). Defaults to
            [0.022, 0.12, 1.04, 0.055, 0.965, 3.0].
        """
        
        self.pars = camb.CAMBparams()
        self.path_to_cp = kwargs.pop('path_to_cp', None)

        if self.path_to_cp:
            self.cp_nn = cp.cosmopower_NN(restore=True, 
                                restore_filename= self.path_to_cp \
                                +'/cosmopower/trained_models/CP_paper/CMB/cmb_TT_NN')

        self.default_params = \
            ['omegabh2', 'omegach2', 'thetaMC', 'tau', 'ns', 'As', 'h']
        self.default_parameter_values = \
                [0.022, 0.12, 1.04, 0.055, 0.965, 3.0, 0.67]

        self.parameters = kwargs.pop('parameters', self.default_params)

        if np.any([p not in self.default_params for p in self.parameters]):
            raise ValueError('Parameters not recognised. Accepted parameters are: ' +
                             ', '.join(self.default_params))

        self.prior_mins = kwargs.pop('prior_mins', 
                [0.005, 0.08, 0.97, 0.01, 0.8, 2.6, 0.5])
        self.prior_maxs = kwargs.pop('prior_maxs', 
                [0.04, 0.21, 1.5, 0.16, 1.2, 3.8, 0.9])
        
        # reorder inputs to expectation based on default
        idx = [self.parameters.index(p) for p in self.default_params 
               if p in self.parameters]
        self.parameters = [self.parameters[i] for i in idx]
        self.prior_mins = [self.prior_mins[i] for i in idx]
        self.prior_maxs = [self.prior_maxs[i] for i in idx]
        
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
        for i in range(len(cube)):
            theta[i] = UniformPrior(self.prior_mins[i], 
                        self.prior_maxs[i])(cube[i])
        return theta
    
    def get_cosmopower_model(self, theta):

        warnings.warn('Note cosmopower ignores thetaMC.')
        # find any missing parameters and insert the default values
        missing = list(sorted(set(self.default_params) - set(self.parameters)))
        missingidx = [self.default_params.index(m) for m in missing][::-1]
        for i in missingidx:
            theta = np.insert(theta, i, self.default_parameter_values[i])
        
        params = {'omega_b': [theta[0]],
                'omega_cdm': [theta[1]],
                'h': [theta[5]],
                'tau_reio': [theta[2]],
                'n_s': [theta[3]],
                'ln10^{10}A_s': [theta[4]],
                }
        
        spectra = self.cp_nn.ten_to_predictions_np(params)[0]*1e12*2.725**2
        return spectra
    
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

        warnings.warn('camb ignores h and uses thetaMC.')

        # find any missing parameters and insert the default values
        missing = list(sorted(set(self.default_params) - set(self.parameters)))
        missingidx = [self.default_params.index(m) for m in missing][::-1]
        for i in missingidx:
            theta = np.insert(theta, i, self.default_parameter_values[i])

        self.pars.set_cosmology(ombh2=theta[0], omch2=theta[1],
                                tau=theta[3], cosmomc_theta=theta[2]/100,
                                theta_H0_range=[5, 1000])
        self.pars.InitPower.set_params(As=np.exp(theta[5])/10**10, ns=theta[4])
        self.pars.set_for_lmax(2500, lens_potential_accuracy=0)

        results = camb.get_background(self.pars) # computes evolution of background cosmology

        cl = results.get_cmb_power_spectra(self.pars, CMB_unit='muK')['total'][:,0]
        cl *= (2*np.pi)/(np.arange(len(cl))*(np.arange(len(cl))+1)) # convert to C_l
        return cl
    
    def get_likelihood(self, data, l, bins, noise=None, cp=False):

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
            if cp:
                cl = self.get_cosmopower_model(theta)
            else:
                cl = self.get_camb_model(theta)
            
            if noise is not None:
                cl += noise

            cl = self.rebin(cl, bins)
            
            # likelihood is a reduced chi squared distribution
            x = (2*l + 1)* data/cl
            logL = -0.5*(-2*chi2(2*l+1).logpdf(x) 
                    - 2*np.log((2*l+1)/cl)).sum()
            
            return logL, []
        return likelihood
    
    def rebin(self, signal, bins):
        indices = bins - 2
        binned_signal = []
        for i in range(len(indices)):
            if indices[i, 0] == indices[i, 1]:
                binned_signal.append(signal[int(indices[i, 0])])
            else:
                binned_signal.append(
                    np.mean(signal[int(indices[i, 0]):int(indices[i, 1])+1]))
        return np.array(binned_signal)
    
    def get_samples(self, l, theta, bins, noise=None, cp=None):

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
        if cp:
            cl =self.get_cosmopower_model(theta)
        else:
            cl = self.get_camb_model(theta)
        
        # if noise then add noise
        if noise is not None:
            cl = cl + noise

        cl = self.rebin(cl, bins)
        noise = self.rebin(noise, bins)

        # draw a sample of (2l+1)*obs/theory from a chi2 distribution
        sample = chi2.rvs(df=2*l + 1, size=len(l))
        sample *= cl # multiply by theory
        sample /= (2*l + 1) # divide by 2l+1

        if noise is not None:
            # because the noise is well known subtract it from the sample
            cl = cl - noise # remove noise from theory
            sample = sample - noise # remove noise from sample
        
        return cl, sample