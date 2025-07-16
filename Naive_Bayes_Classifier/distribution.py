import numpy as np
from scipy.special import erfinv
from scipy.stats import binom

class Distribution:
    
    def estimate_gaussian_params(sample):
    ### START CODE HERE ###
        mu = np.mean(sample)
        sigma = np.std(sample)
        ### END CODE HERE ###

        return mu, sigma

    def estimate_binomial_params(sample):
        ### START CODE HERE ###
        n = 30
        p = (sample / n).mean()
        ### END CODE HERE ###

        return n, p
    
    def estimate_uniform_params(sample):
        ### START CODE HERE ###
        a = sample.min()
        b = sample.max()
        ### END CODE HERE ###

        return a, b
    
    
    def uniform_generator(self,a,b,num_sample=100):
    #Generates an array of uniformly distributed random numbers within the given range
        np.random.seed(42)
        array = np.random.uniform(a,b,num_sample)
        return array
    
    def _inverse_cdf_gaussian(self, y, mu, sigma):
    # Calculates the inverse cumulative distribution function of a Gaussian function.
        x = mu + erfinv(2*y - 1) * sigma*(2**0.5)
        return x

    def gaussian_generator(self, mu, sigma, num_samples):
        u = self.uniform_generator(0,1,num_samples)
        array = self._inverse_cdf_gaussian(u, mu, sigma)
        return array 
    
    def _inverse_cdf_binomial(self,y,n,p):
        """Calculate the inverse cumulative distribution function of a binomial distribution

        Args:
            y (float or np.ndarray): The probability or array of probabilities
            n (int): The number of trials in the binomial distribution
            p (float): The probability of success in each trail
        """
        x = binom.ppf(y,n,p)
        return x

    def binomial_generator(self,n,p,num_samples):
        u = self.uniform_generator(0,1,num_samples)
        array = self._inverse_cdf_binomial(u,n,p)
        return array
    
    def pdf_uniform(self,x,a,b):
        """Calculates the probability density function (PDF) for a uniform distribution 
        between 'a' and 'b' at a given point 'x'.

        Args:
            x (float): The value at which the PDF is evaluated.
            a (float): The lower bound of the uniform distribution.
            b (float): The upper bound of the uniform distribution.
            
        Returns:
            float: The PDF value at the given point 'x'. Returns 0 if 'x' is outside the range [a,b]
        """
        pdf = (0) if (x<a or x>b) else 1/(b-a)
        return pdf
    
    def pdf_gaussian(self,x,mu,sigma):
        """Calculate the PDF of a gaussian distribution at a given value. 

        Args:
            x (float or array-like): The value at which to evaluate pdf
            mu (float): The mean of the Gaussian Distribution
            sigma (float): The standard deviation of the Gaussian Distribution.
            
        Returns:
            float or ndarray: The PDF value(s) at the given point(s) x.
        """
        
        coefficient = 1.0 / (sigma * np.sqrt(2*np.pi))
        exponent = -0.5*((x-mu)/sigma)**2
        pdf = coefficient * np.exp(exponent)
        
        return pdf
    
    def pdf_binomial(self,x,n,p):
        """
        For binomial distribution, since it is a discrete distribution,
        we will be using the PMF instead of the PDF.
        Calculate the Probability Mass Function (PMF) of a binomial distribution at a specific value.
    

        Args:
            x (int): The value at which to evaluate the PMF
            n (int): The number of trials in the binomial distribution
            p (float): The probability of success for each trail
        
        Returns:
            float: The probability mass function (PMF) of the binomial distribution at the specified value.
        """
        pdf = binom.pmf(x, n, p)
        return pdf
    