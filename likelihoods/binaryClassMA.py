# Copyright (c) 2018 Pablo Moreno-Munoz
# Universidad Carlos III de Madrid and University of Sheffield

import sys
import numpy as np
from GPy.likelihoods import link_functions
from GPy.likelihoods import Likelihood
from GPy.util.misc import safe_exp, safe_square
from GPy.util.univariate_Gaussian import std_norm_pdf, std_norm_cdf
from scipy.special import logsumexp


class BinaryClassMA(Likelihood):
    """
    Bernoulli likelihood with a latent function over its parameter

    """

    def __init__(self, gp_link=None):
        if gp_link is None:
            gp_link = link_functions.Identity()

        super(BinaryClassMA, self).__init__(gp_link, name='BinaryClassMA')

    def pdf(self, f, y, Y_metadata=None):
        ef = safe_exp(f)
        p = ef / (1 + ef)
        pdf = p ** (y) * (1 - p) ** (1 - y)
        return pdf

    def logpdf(self, f, y, Y_metadata=None):
        ef = safe_exp(f)
        p = ef / (1 + ef)
        p = np.clip(p, 1e-9, 1. - 1e-9) #numerical stability
        logpdf = (y * np.log(p)) + ((1 - y) * np.log(1 - p))
        return logpdf

    def mean(self, f, Y_metadata=None):
        ef = safe_exp(f)
        p = ef / (1 + ef)
        p = np.clip(p, 1e-9, 1. - 1e-9) #numerical stability
        mean = p
        return mean

    def mean_sq(self, f, Y_metadata=None):
        ef = safe_exp(f)
        p = ef / (1 + ef)
        p = np.clip(p, 1e-9, 1. - 1e-9) #numerical stability
        mean_sq = np.square(p)
        return mean_sq

    def variance(self, f, Y_metadata=None):
        ef = safe_exp(f)
        p = ef / (1 + ef)
        p = np.clip(p, 1e-9, 1. - 1e-9) #numerical stability
        var = p*(1 - p)
        return var

    def samples(self, f ,num_samples, Y_metadata=None):
        ef = safe_exp(f)
        p = ef / (1 + ef)
        p = np.clip(p, 1e-9, 1. - 1e-9)  # numerical stability
        samples = np.random.binomial(n=1, p=p)
        return samples

    def dlogp_df(self, f, y, Y_metadata=None):
        ef = safe_exp(f)
        p = ef / (1 + ef)
        p = np.clip(p, 1e-9, 1. - 1e-9)  # numerical stability
        dlogp = ((y - p) / (1 - p)) * (1 / (1 + ef))
        #dlogp = ( y - (1 - y)*(p / (1 - p)))*(1 / (1 + ef))
        return dlogp

    def d2logp_df2(self, f, y, Y_metadata=None):
        ef = safe_exp(f)
        p = ef / (1 + ef)
        p = np.clip(p, 1e-9, 1. - 1e-9)  # numerical stability
        d2logp = - p / (1 + ef)
        #d2logp = (1 / (1 + ef))*((p*(y - 1)/((1 + ef)*(1 - p)**2)) - p*y + (1 - y)*(p**2 / (1 - p)))
        return d2logp
    
    def KappaFuncEx(self, m, v):
        kapp = 1/np.sqrt(1 + np.pi*v/8)
        a = kapp*m
        Eq_g = 1/(1 + safe_exp(-a))
        return Eq_g
    
    def logisticFunc(self, a):
        logit = 1/(1 + safe_exp(-a))
        return logit

    def var_exp(self, Y, m, v, GN=None, gh_points=None, Y_metadata=None):
        # Variational Expectation
        # gh: Gaussian-Hermite quadrature
        if gh_points is None:
            gh_f, gh_w = self._gh_points()
        else:
            gh_f, gh_w = gh_points
        
        N, NM = Y.shape
        iAnn = Y_metadata
        m_f, m_g = m[:, :1], m[:, 1:]
        v_f, v_g = v[:, :1], v[:, 1:]
        
        ## for the Gauss-Hermite quadrature
        gh_w = gh_w / np.sqrt(np.pi)
        
        # E_{q(g_m^m)}[z_n^m] --> Bishop (4.153, 4.154)
        Eq_g = np.empty((m.shape[0],NM))
        for r in range(NM):
            alpha_im = gh_f[None, :] * np.sqrt(2. * v_g[:,r:r+1]) + m_g[:,r:r+1]
            zn = self.logisticFunc(alpha_im)
            Eq_g[:,r] = zn.dot(gh_w[:,None]).flatten()
        # Eq_g = self.KappaFuncEx(m_g, v_g)
        # Eq_g = np.clip(Eq_g, 1e-9, 1. - 1e-9) #numerical stability
        
        ## E_{q(f_n)}[p_n] --> Gauss-Hermite quadrature
        x = gh_f[None, :] * np.sqrt(2. * v_f) + m_f
        logp = np.log(self.logisticFunc(x))
        Eq_f1 = logp.dot(gh_w[:,None])
        # Eq_f1 = np.clip(Eq_f1, 1e-9, 1. - 1e-9) #numerical stability
        
        ## E_{q(f_n)}[1-p_n] --> Gauss-Hermite quadrature
        logp1 = np.log(self.logisticFunc(-x))
        Eq_f2 = logp1.dot(gh_w[:,None])
        
        #Variational Expectation ##########################################
        var_exp = np.sum( ( Y*Eq_g*Eq_f1 + (1-Y)*Eq_g*Eq_f2 + np.log(0.5)*(1-Eq_g) )*iAnn, 1).reshape((N,1))
        return var_exp

    def var_exp_derivatives(self, Y, m, v, GN=None, gh_points=None, Y_metadata=None):
        # Variational Expectations of derivatives
        # gh: Gaussian-Hermite quadrature
        if gh_points is None:
            gh_f, gh_w = self._gh_points()
        else:
            gh_f, gh_w = gh_points
        
        N, NM = Y.shape
        iAnn = Y_metadata
        m_f, m_g = m[:, :1], m[:, 1:]
        v_f, v_g = v[:, :1], v[:, 1:]
        
        var_exp_dm = np.empty((m.shape[0],NM+1))
        var_exp_dv = np.empty((m.shape[0],NM+1)) 
        
        ## for the Gauss-Hermite quadrature
        gh_w = gh_w / np.sqrt(np.pi)
        
        # E_{q(g_m^m)}[z_n^m] --> Bishop (4.153, 4.154)
        Eq_g = np.empty((m.shape[0],NM))
        for r in range(NM):
            alpha_im = gh_f[None, :] * np.sqrt(2. * v_g[:,r:r+1]) + m_g[:,r:r+1]
            zn = self.logisticFunc(alpha_im)
            Eq_g[:,r] = zn.dot(gh_w[:,None]).flatten()
        # Eq_g = self.KappaFuncEx(m_g, v_g)
        # Eq_g = np.clip(Eq_g, 1e-9, 1. - 1e-9) #numerical stability
        
        ## E_{q(f_n)}[p_n] --> Gauss-Hermite quadrature
        x = gh_f[None, :] * np.sqrt(2. * v_f) + m_f
        logp = np.log(self.logisticFunc(x))
        Eq_f1 = logp.dot(gh_w[:,None])
        # Eq_f1 = np.clip(Eq_f1, 1e-9, 1. - 1e-9) #numerical stability
        
        ## E_{q(f_n)}[1-p_n] --> Gauss-Hermite quadrature
        logp1 = np.log(self.logisticFunc(-x))
        Eq_f2 = logp1.dot(gh_w[:,None])
            
        
        ## E_{q(f_n)}[d2fn2] --> Gauss-Hermite quadrature
        pn1_pn = (1/(1 + safe_exp(-x)))*(1/(1 + safe_exp(x)))
        E_qf_pn1_pn = -0.5*pn1_pn.dot(gh_w[:,None])
        
        # E_{q(f_n)}[d_logPn_dfn] --> Gauss-Hermite quadrature
        dlogp_dfn = self.logisticFunc(-x)
        E_dlogp_dfn = dlogp_dfn.dot(gh_w[:,None])
        
        # E_{q(f_n)}[d_logPn1_dfn] --> Gauss-Hermite quadrature
        dlogp1_dfn = self.logisticFunc(x)
        E_dlogp1_dfn = -dlogp1_dfn.dot(gh_w[:,None])
        
        # E_{q(f_n)}[d2_logPn_dfn2] --> Gauss-Hermite quadrature
        d2logp1_dfn2_1 = self.logisticFunc(x)
        d2logp1_dfn2_2 = self.logisticFunc(x)**2
        E_d2logp_dfn2 = -0.5*( d2logp1_dfn2_1.dot(gh_w[:,None]) - d2logp1_dfn2_2.dot(gh_w[:,None]))
        
        var_exp_dm[:, 0] = np.sum( ( Y*Eq_g*E_dlogp_dfn + (1-Y)*Eq_g*E_dlogp1_dfn )*iAnn, 1)
        var_exp_dv[:, 0] = np.sum( (Y*Eq_g*E_d2logp_dfn2 + (1-Y)*Eq_g*E_d2logp_dfn2)*iAnn, 1)
        
        Const = Y*Eq_f1 + (1-Y)*Eq_f2 + np.log(2)
        for r in range(NM):
            alpha_im = gh_f[None, :] * np.sqrt(2. * v_g[:,r:r+1]) + m_g[:,r:r+1]
            znm  = self.logisticFunc(alpha_im)
            znm2 = self.logisticFunc(alpha_im)**2
            znm3 = self.logisticFunc(alpha_im)**3
            Eznm  = znm.dot(gh_w[:,None])
            Eznm2 = znm2.dot(gh_w[:,None])
            Eznm3 = znm3.dot(gh_w[:,None])
            
            
            var_exp_dm[:, r+1] = ((Eznm - Eznm2)*Const[:,r:r+1]*iAnn[:,r:r+1]).flatten()
            
            
            var_exp_dv[:, r+1] = (0.5*(2*Eznm3 - 3*Eznm2 + Eznm)*Const[:,r:r+1]*iAnn[:,r:r+1]).flatten()
        
        return var_exp_dm, var_exp_dv

    # def predictive(self, m, v, gh_points=None, Y_metadata=None):
    #     # Variational Expectation
    #     # gh: Gaussian-Hermite quadrature
    #     if gh_points is None:
    #         gh_f, gh_w = self._gh_points()
    #     else:
    #         gh_f, gh_w = gh_points
    #
    #     gh_w = gh_w / np.sqrt(np.pi)
    #     m, v= m.flatten(), v.flatten()
    #     f = gh_f[None, :] * np.sqrt(2. * v[:, None]) + m[:, None]
    #     mean = self.mean(f)
    #     var = self.variance(f).dot(gh_w[:,None]) + self.mean_sq(f).dot(gh_w[:,None]) - np.square(mean.dot(gh_w[:,None]))
    #     mean_pred = mean.dot(gh_w[:,None])
    #     var_pred = var
    #     return mean_pred, var_pred

    def predictive(self, m, v,Y_metadata=None):
        # This can be checked in eq 4.152 Pattern recognition Bishop
        # with lambda = 1
        #mean_pred = std_norm_cdf(m / np.sqrt(1 + v))  #Here the mean prediction is already influenced by the variance v
        
        # a = m/np.sqrt(1 + np.pi*v/8)
        _, R = m.shape
        mean_pred = []
        var_pred = []
        mean_pred.append(m[:,0,None])
        var_pred.append(v[:,0,None])
        for r in range(R-1):
            auxm = m[:,r+1,None]
            auxv = v[:,r+1,None]
            mean_pred.append(auxm)
            var_pred.append(auxv)
        return mean_pred, var_pred                    #of the prediciton, so with don't need any confidence variance

    def log_predictive(self, Ytest, mu_F_star, v_F_star, num_samples):
        Ntest, D = mu_F_star.shape
        F_samples = np.empty((Ntest, num_samples, D))
        # function samples:
        for d in range(D):
            mu_fd_star = mu_F_star[:, d][:, None]
            var_fd_star = v_F_star[:, d][:, None]
            F_samples[:, :, d] = np.random.normal(mu_fd_star, np.sqrt(var_fd_star), size=(Ntest, num_samples))

        # monte-carlo:
        log_pred = -np.log(num_samples) + logsumexp(self.logpdf(F_samples[:,:,0], Ytest), axis=-1)
        log_pred = np.array(log_pred).reshape(*Ytest.shape)
        "I just changed this to have the log_predictive of each data point and not a mean values"
        #log_predictive = (1/num_samples)*log_pred.sum()

        return log_pred

    def get_metadata(self, Y):
        _, R = Y.shape
        dim_y = 1
        dim_f = R + 1
        dim_p = 1
        return dim_y, dim_f, dim_p

    def ismulti(self):
        # Returns if the distribution is multivariate
        return False
