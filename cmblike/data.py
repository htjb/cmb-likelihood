import numpy as np

class get_data():
    def __init__(self, base_dir='data/'):
        self.base_dir = base_dir
    
    def get_planck(self):

        """
        Function to load in the planck power spectrum data.

        Returns
        -------
        p: power spectrum
        l: the multipoles
        """

        tt = np.loadtxt(self.base_dir + 
                        'TT_power_spec.txt', delimiter=',', dtype=str)

        l, p = [], []
        for i in range(len(tt)):
            if tt[i][0] == 'Planck binned      ':
                l.append(tt[i][2].astype('float')) # ell
                p.append(tt[i][4].astype('float')) # power spectrum
        p, l = np.array(p), np.array(l)
        p *= (2*np.pi)/(l*(l+1)) # convert to C_l
        return p, l

    def get_wmap(self):
        
        """
        Function to load in the wmap power spectrum data.

        Returns
        -------
        p: power spectrum
        l: the multipoles
        """
        wmap = np.loadtxt(self.base_dir + 
                          'wmap_binned.txt')
        l = wmap[:, 0]
        p = wmap[:, 3]
        return p, l