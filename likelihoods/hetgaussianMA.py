# Copyright (c) 2018 Pablo Moreno-Munoz
# Universidad Carlos III de Madrid and University of Sheffield

import sys
import numpy as np
from GPy.likelihoods import link_functions
from GPy.likelihoods import Likelihood
from GPy.util.misc import safe_exp, safe_square
from scipy.special import logsumexp

class HetGaussianMA(Likelihood):
    """
    Heterocedastic Gaussian likelihood with a latent function over its parameter

    """

    def __init__(self, gp_link=None):
        if gp_link is None:
            gp_link = link_functions.Identity()

        super(HetGaussianMA, self).__init__(gp_link, name='HetGaussianMA')

    def pdf(self, F, y, Y_metadata):
        N, _ = y.shape
        iAnn = Y_metadata
        var_r = safe_exp(F[:,1:])
        ym = (y - F[:,0])
        logpdf_m = -0.5( np.log(2 * np.pi * var_r) - ( (ym**2) / var_r))
        logpdf = np.sum(logpdf_m*iAnn,1)
        pdf = safe_exp(logpdf).reshape((N,1))
        return pdf

    def logpdf(self, F, y, Y_metadata=None):
        N, _ = y.shape
        iAnn = Y_metadata
        var_r = safe_exp(F[:,1:])
        ym = (y - F[:,0])
        logpdf_m = -0.5( np.log(2 * np.pi * var_r) - ( (ym**2) / var_r))
        logpdf = np.sum(logpdf_m*iAnn,1)
        return logpdf

    def logpdf_sampling(self, F, y, Y_metadata=None):
        e_var = safe_exp(F[:,1,:])
        ym =(np.tile(y, (1,F.shape[2])) - F[:,0,:])
        logpdf = -0.5*np.log(2*np.pi) - (0.5*F[:,1]) - 0.5*((safe_square(ym)) / e_var)
        return logpdf

    def samples(self, F , num_samples, Y_metadata=None):
        e_var = safe_exp(F[:,1])
        samples = np.random.normal(loc=F[:,0], scale=np.sqrt(e_var))
        return samples[:,None]

    def var_exp(self, Y, M, V, GN=None, gh_points=None, Y_metadata=None):
        # Variational Expectation (Analytical)
        # E_q(fid)[log(p(yi|fid))]
        N, _ = Y.shape
        iAnn = Y_metadata
        m_fmean, m_fvar = M[:, :1], M[:, 1:]
        v_fmean, v_fvar = V[:, :1], V[:, 1:]
        precision = safe_exp(- m_fvar + (0.5*v_fvar))
        precision = np.clip(precision, -1e9, 1e9)  # numerical stability
        squares = (safe_square(Y) + safe_square(m_fmean) + v_fmean - (2*m_fmean*Y))
        squares = np.clip(squares, -1e9, 1e9)  # numerical stability
        var_exp = np.sum((- (0.5*np.log(2 * np.pi)) - (0.5*m_fvar) - (0.5*precision*squares))*iAnn, axis=1).reshape((N,1))
        return var_exp

    def var_exp_derivatives(self, Y, M, V, GN=None, gh_points=None, Y_metadata=None):
        # Variational Expectations of derivatives
        N, R = Y.shape
        iAnn = Y_metadata
        
        var_exp_dm = np.empty((M.shape[0],R+1))
        var_exp_dv = np.empty((M.shape[0],R+1))

        m_fmean, m_fvar = M[:, :1], M[:, 1:]
        v_fmean, v_fvar = V[:, :1], V[:, 1:]
        precision = safe_exp(- m_fvar + (0.5*v_fvar))
        precision = np.clip(precision, -1e9, 1e9)  # numerical stability
        squares = (np.square(Y) + np.square(m_fmean) + v_fmean - (2*m_fmean*Y))
        squares = np.clip(squares, -1e9, 1e9)  # numerical stability
        var_exp_dm[:, 0] = np.sum(precision*(Y - m_fmean)*iAnn, axis = 1)
        var_exp_dm[:, 1:] = 0.5*((precision * squares) - 1.)*iAnn
        var_exp_dv[:, 0] = np.sum(-0.5*precision*iAnn, 1)
        var_exp_dv[:, 1:] = -0.25*precision*squares*iAnn
        return var_exp_dm, var_exp_dv

    def predictive(self, M, V, gh_points=None, Y_metadata=None):
        # Variational Expectation
        # gh: Gaussian-Hermite quadrature
        if gh_points is None:
            gh_f, gh_w = self._gh_points()
        else:
            gh_f, gh_w = gh_points

        gh_w = gh_w / np.sqrt(np.pi)
        # f1 = gh_f[None, :] * np.sqrt(2. * V[:,0,None]) + M[:,0,None]
        # f2 = gh_f[None, :] * np.sqrt(2. * V[:,1,None]) + M[:,1,None]
        
        _, R = M.shape
        mean_pred = []
        var_pred = []
        mean_pred.append(M[:,0,None])
        var_pred.append(V[:,0,None])
        for m in range(R-1):
            auxm = safe_exp(M[:,m+1,None] + V[:,m+1,None]/2)
            auxv = (safe_exp(V[:,m+1,None]) - 1)*safe_exp(2*M[:,m+1,None] + V[:,m+1,None])
            mean_pred.append(auxm)
            var_pred.append(auxv)
        
        return mean_pred, var_pred

    def log_predictive(self, Ytest, mu_F_star, v_F_star, num_samples):
        Ntest, D = mu_F_star.shape
        F_samples = np.empty((Ntest, D, num_samples))
        # function samples:
        for d in range(D):
            mu_fd_star = mu_F_star[:, d][:, None]
            var_fd_star = v_F_star[:, d][:, None]
            F_samples[:, d, :] = np.random.normal(mu_fd_star, np.sqrt(var_fd_star), size=(Ntest, num_samples))

        # monte-carlo:
        log_pred = -np.log(num_samples) + logsumexp(self.logpdf_sampling(F_samples, Ytest), axis=-1)
        log_pred = np.array(log_pred).reshape(*Ytest.shape)
        "I just changed this to have the log_predictive of each data point and not a mean values"
        #log_predictive = (1/num_samples)*log_pred.sum()

        return log_pred

    def get_metadata(self, iAnn):
        _, R = iAnn.shape
        dim_y = 1
        dim_f = R + 1
        dim_p = 1
        return dim_y, dim_f, dim_p

    def ismulti(self):
        # Returns if the distribution is multivariate
        return False
