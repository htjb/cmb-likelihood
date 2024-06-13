import numpy as np
import camb
from scipy.stats import chi2
import cosmopower as cp
from cmblike.noise import planck_noise, wmap_noise
import yaml

class CMB():
    def __init__(self, config, **kwargs):

        """
        Class to generate CMB power spectra from CAMB or cosmopower,
        build and sample a likelihood function for a given data set and noise.

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
        
        if not type(config) == dict:
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)

        if config['path_to_cp']:
            self.cp_nn = cp.cosmopower_NN(restore=True, 
                                restore_filename= config['path_to_cp'] \
                                +'/cosmopower/trained_models/CP_paper/CMB/cmb_TT_NN')
        else:
            self.pars = camb.CAMBparams()
    
    def get_cosmopower_model(self):
        
        params = {'omega_b': [float(self.config['parameters']['omegabh2'])],
                'omega_cdm': [float(self.config['parameters']['omegach2'])],
                'h': [float(self.config['parameters']['h'])],
                'tau_reio': [float(self.config['parameters']['tau'])],
                'n_s': [float(self.config['parameters']['ns'])],
                'ln10^{10}A_s': [float(self.config['parameters']['As'])],
                }
        
        spectra = self.cp_nn.ten_to_predictions_np(params)[0]*1e12*2.725**2
        return spectra
    
    def get_camb_model(self):

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

        self.pars.set_cosmology(ombh2=float(self.config['parameters']['omegabh2']),
                omch2=float(self.config['parameters']['omegach2']),
                tau=float(self.config['parameters']['tau']), 
                cosmomc_theta=float(self.config['parameters']['thetaMC'])/100,
                theta_H0_range=[5, 1000])
        self.pars.InitPower.set_params(
            As=np.exp(float(self.config['parameters']['As']))/10**10, 
            ns=float(self.config['parameters']['ns']))
        self.pars.set_for_lmax(2500, lens_potential_accuracy=0)

        # computes evolution of background cosmology
        results = camb.get_background(self.pars)

        cl = results.get_cmb_power_spectra(self.pars, 
                                           CMB_unit='muK')['total'][:,0]
        # convert to C_l
        cl *= (2*np.pi)/(np.arange(len(cl))*(np.arange(len(cl))+1))
        return cl
    
    def get_samples(self, l, theta, bins, noise=None, cp=None):

        """
        Code to generate observations of a theoretical CMB power spectrum.

        Parameters
        ----------
        theta: array
            The cosmological parameters as generated from self.prior.

        bins: array
            The bin edges for the data.

        noise: array
            The noise associated with the data. If None then no noise is added.
            Assumes that the noise is at every l between 2 and 2508.
        
        cp: bool
            Whether to use cosmopower or not. Defaults to False.
        """
        if cp:
            cl =self.get_cosmopower_model(theta)
        else:
            cl = self.get_camb_model(theta)
        
        # if noise then add noise
        if noise is not None:
            cl = cl + noise

        l = np.arange(0, len(cl))+2

        # draw a sample of (2l+1)*obs/theory from a chi2 distribution
        sample = chi2.rvs(df=2*l + 1, size=len(l))
        sample *= cl # multiply by theory
        sample /= (2*l + 1) # divide by 2l+1

        cl = self.rebin(cl, bins)
        sample = self.rebin(sample, bins)
        
        return cl, sample