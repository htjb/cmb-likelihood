import numpy as np

class cmb_noise():
    def __init__(self, l, theta, sigma_T):
        
        """
        Function to calcualte the noise power spectrum for a given
        beam and sensitivity. This noise
        is associated with the resolution of the instrument.

        Parameters
        ----------
        l: array
            The multipoles.
        
        theta: array
            The beam size in radians for each frequency.
        
        sigma_T: array
            The sensitivity in microK radians for each frequency.
        
        Returns
        -------
        noise: array
            The noise power at each multipole.
        """
    
        self.l = l
        self.theta = theta
        self.sigma_T = sigma_T

    def calculate_noise(self):
        # calculation from https://arxiv.org/abs/1612.08270
        nis = []
        for i in range(len(self.sigma_T)):
            ninst = self.sigma_T[i]**-2 * \
                np.exp(-self.l*(self.l+1)*self.theta[i]**2/
                       (8*np.log(2))) #one over ninst
            nis.append(ninst)
        ninst = np.array(nis).T
        ninst = np.sum(ninst, axis=1)
        noise = 1/ninst

        return noise, self.l

class planck_noise(cmb_noise):
    def __init__(self, l=np.arange(2, 2500)):
        #table 4 in https://arxiv.org/pdf/1807.06205.pdf
        theta= np.array([9.66, 7.22, 4.90])*np.pi/60/180
        sigma_T=np.array([1.29, 0.55, 0.78])*np.pi/180
        super().__init__(l, theta, sigma_T)
    
class wmap_noise(cmb_noise):
    def __init__(self, l=np.arange(2, 2500)):
        theta= np.array([0.82, 0.62, 0.49, 0.33, 0.21])*np.pi/180
        #taken from https://wmap.gsfc.nasa.gov/mission/observatory_sens.html#:~:text=WMAP%20Sensitivity,x%200.3°%20square%20pixel.
        sigma_T=np.array([35, 35, 35, 35, 35])*(0.3)**2*np.pi/180
        super().__init__(l, theta, sigma_T)


