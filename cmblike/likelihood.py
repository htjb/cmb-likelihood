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