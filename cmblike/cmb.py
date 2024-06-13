import numpy as np
import camb
from scipy.stats import chi2
import cosmopower as cp
from cmblike.noise import planck_noise, wmap_noise
import yaml

class CMB():
    def __init__(self, **kwargs):

        """
        Class to generate CMB power spectra from CAMB or cosmopower,
        build and sample a likelihood function for a given data set and noise.

        Parameters:
        -----------
        path_to_cp: str
            The path to the cosmopower directory. Defaults to None.
        """
        
        self.path_to_cp = kwargs.get('path_to_cp', None)

        if self.path_to_cp:
            self.cp_nn = cp.cosmopower_NN(restore=True, 
                    restore_filename= self.path_to_cp \
                    +'/cosmopower/trained_models/CP_paper/CMB/cmb_TT_NN')
        else:
            self.pars = camb.CAMBparams()
    
    def get_cosmopower_model(self, parameters):

        """
        Function to get the cosmopower model for the CMB power spectrum.

        Parameters:
        -----------
        theta: dictionary
            An dictionary of cosmological parameters. For cosmopower the
            required parameters are 'omegabh2', 'omegach2', 'h', 
            'tau', 'ns', 'As'.
        
        Returns:
        --------
        spectra: array
            The CMB power spectrum at l=2 to 2508.
        """
        
        params = {'omega_b': [parameters['omegabh2']],
                'omega_cdm': [parameters['omegach2']],
                'h': [parameters['h']],
                'tau_reio': [parameters['tau']],
                'n_s': [parameters['ns']],
                'ln10^{10}A_s': [parameters['As']],
                }
        
        spectra = self.cp_nn.ten_to_predictions_np(params)[0]*1e12*2.725**2
        return spectra
    
    def get_camb_model(self, parameters):

        """
        Function to get the CAMB model for the CMB power spectrum.

        Parameters
        ----------
        theta: dictionary
            An dictionary of cosmological parameters. For CAMB the
            required parameters are 'omegabh2', 'omegach2', 'thetaMC', 
            'tau', 'ns', 'As'.
        
        Returns:
        --------
        cl: array
            The CMB power spectrum at l=2 to 2508.
        """

        self.pars.set_cosmology(ombh2=parameters['omegabh2'],
                omch2=parameters['omegach2'],
                tau=parameters['tau'], 
                cosmomc_theta=parameters['thetaMC']/100,
                theta_H0_range=[5, 1000])
        self.pars.InitPower.set_params(
            As=np.exp(parameters['As'])/10**10, 
            ns=parameters['ns'])
        self.pars.set_for_lmax(2500, lens_potential_accuracy=0)

        # computes evolution of background cosmology
        results = camb.get_background(self.pars)

        cl = results.get_cmb_power_spectra(self.pars, 
                                           CMB_unit='muK')['total'][:,0]
        # convert to C_l
        cl *= (2*np.pi)/(np.arange(len(cl))*(np.arange(len(cl))+1))

        cl = np.interp(np.arange(2, 2509), np.arange(0, len(cl)), cl)
        return cl
    
    def get_samples(self, theta, bins, noise_profile):

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