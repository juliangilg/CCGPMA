# Copyright (c) 2018 Pablo Moreno-Munoz
# Universidad Carlos III de Madrid and University of Sheffield

import sys
import numpy as np
from GPy.likelihoods import link_functions
from GPy.likelihoods import Likelihood
from GPy.util.misc import safe_exp, safe_square
from GPy.util.univariate_Gaussian import std_norm_pdf, std_norm_cdf
from functools import reduce # only in Python 3
from scipy.special import logsumexp


class MultiClassMA(Likelihood):
    """
    Bernoulli likelihood with a latent function over its parameter

    """

    def __init__(self, Y, gp_link=None):
        if gp_link is None:
            gp_link = link_functions.Identity()
            self.Y = Y
            Clases = np.unique(self.Y)
            auxC = np.prod(Clases)
            if auxC > 0:
                K = len(np.unique(self.Y))
            else:
                K = len(np.unique(self.Y)) - 1
            
            self.K = K

        super(MultiClassMA, self).__init__(gp_link, name='MultiClassMA')

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
    
    def softmax(self, A):
        num = safe_exp(A)
        den = np.sum(num, 1)
        den = den[:,np.newaxis]
        zeta_k = num/den
        return zeta_k
    
    def one_of_K(self, y, K):
        Yhat = np.ones((y.shape[0], K))
        for k in range(K):
            Yhat[:,k,None] = (y==k+1).astype(np.int)
        return Yhat
    
    def GaussHermiteMC(self, gh_f, gh_w, m_f, v_f, K):
        N = m_f.shape[0]
        if K <= 3:
            expanded_F_tuples = []
            grid_tuple = [m_f.shape[0]]
            for k in range(K):
                grid_tuple.append(gh_f.shape[0])
                expanded_fd_tuple = [1]*(K+1)
                expanded_fd_tuple[k+1] = gh_f.shape[0]
                expanded_F_tuples.append(tuple(expanded_fd_tuple))
                
            # mean-variance tuple
            mv_tuple = [1]*(K+1)
            mv_tuple[0] = m_f.shape[0]
            mv_tuple = tuple(mv_tuple)
        
            # building, normalizing and reshaping the grids
            F = np.zeros((reduce(lambda x, y: x * y, grid_tuple),K))
            for d in range(K):
                fd = np.zeros(tuple(grid_tuple))
                fd[:] = np.reshape(gh_f, expanded_F_tuples[d])*np.sqrt(2*np.reshape(v_f[:,d],mv_tuple)) \
                        + np.reshape(m_f[:,d],mv_tuple)
                F[:,d,None] = fd.reshape(reduce(lambda x, y: x * y, grid_tuple), -1, order='C')
                
            log_zeta = np.log(self.softmax(F))
            var_exp_log_zeta = np.zeros((N,K))
            for k in range(K):
                log_zetak = log_zeta[:,k].reshape(tuple(grid_tuple))
                var_exp_k = log_zetak.dot(gh_w) / np.sqrt(np.pi)
                for kl in range(K-1):
                    var_exp_k = var_exp_k.dot(gh_w) / np.sqrt(np.pi)
                var_exp_log_zeta[:,k] = var_exp_k
                
            zeta = (self.softmax(F))
            var_exp_zeta = np.zeros((N,K))
            for k in range(K):
                zetak = zeta[:,k].reshape(tuple(grid_tuple))
                var_exp_k1 = zetak.dot(gh_w) / np.sqrt(np.pi)
                for kl in range(K-1):
                    var_exp_k1 = var_exp_k1.dot(gh_w) / np.sqrt(np.pi)
                var_exp_zeta[:,k] = var_exp_k1
                
            zeta2 = (self.softmax(F))**2
            var_exp_zeta2 = np.zeros((N,K))
            for k in range(K):
                zetak2 = zeta2[:,k].reshape(tuple(grid_tuple))
                var_exp_k2 = zetak2.dot(gh_w) / np.sqrt(np.pi)
                for kl in range(K-1):
                    var_exp_k2 = var_exp_k2.dot(gh_w) / np.sqrt(np.pi)
                var_exp_zeta2[:,k] = var_exp_k2
                
        else:
            var_exp_log_zeta = np.zeros((N,K))
            var_exp_zeta = np.zeros((N,K))
            var_exp_zeta2 = np.zeros((N,K))
            
            gh_f, gh_w = np.polynomial.hermite.hermgauss(15)
            expanded_F_tuples = []
            grid_tuple = [1]
            for k in range(K):
                grid_tuple.append(gh_f.shape[0])
                expanded_fd_tuple = [1]*(K+1)
                expanded_fd_tuple[k+1] = gh_f.shape[0]
                expanded_F_tuples.append(tuple(expanded_fd_tuple))
                
            # mean-variance tuple
            mv_tuple = [1]*(K+1)
            mv_tuple[0] = 1
            mv_tuple = tuple(mv_tuple)
            
            # building, normalizing and reshaping the grids
            for i in range(N):
                Faux = np.zeros((reduce(lambda x, y: x * y, grid_tuple),K))
                for d in range(K):
                    fd = np.zeros(tuple(grid_tuple))
                    fd[:] = np.reshape(gh_f, expanded_F_tuples[d])*np.sqrt(2*np.reshape(v_f[i,d],mv_tuple)) \
                            + np.reshape(m_f[i,d],mv_tuple)
                    Faux[:,d,None] = fd.reshape(reduce(lambda x, y: x * y, grid_tuple), -1, order='C')
                    
                log_zeta = np.log(self.softmax(Faux))
                for k in range(K):
                    log_zetak = log_zeta[:,k].reshape(tuple(grid_tuple))
                    var_exp_k = log_zetak.dot(gh_w) / np.sqrt(np.pi)
                    for kl in range(K-1):
                        var_exp_k = var_exp_k.dot(gh_w) / np.sqrt(np.pi)
                    var_exp_log_zeta[i,k] = var_exp_k
                
                zeta = (self.softmax(Faux))
                for k in range(K):
                    zetak = zeta[:,k].reshape(tuple(grid_tuple))
                    var_exp_k1 = zetak.dot(gh_w) / np.sqrt(np.pi)
                    for kl in range(K-1):
                        var_exp_k1 = var_exp_k1.dot(gh_w) / np.sqrt(np.pi)
                    var_exp_zeta[i,k] = var_exp_k1
                
                zeta2 = (self.softmax(Faux))**2
                for k in range(K):
                    zetak2 = zeta2[:,k].reshape(tuple(grid_tuple))
                    var_exp_k2 = zetak2.dot(gh_w) / np.sqrt(np.pi)
                    for kl in range(K-1):
                        var_exp_k2 = var_exp_k2.dot(gh_w) / np.sqrt(np.pi)
                    var_exp_zeta2[i,k] = var_exp_k2
                
        
        return var_exp_log_zeta, var_exp_zeta, var_exp_zeta2

    def var_exp(self, Y, m, v, GN=None, gh_points=None, Y_metadata=None):
        # Variational Expectation
        # gh: Gaussian-Hermite quadrature
        if gh_points is None:
            gh_f, gh_w = self._gh_points()
        else:
            gh_f, gh_w = gh_points
        
        N, R = Y.shape
        Clases = np.unique(Y)
        auxC = np.prod(Clases)
        if auxC > 0:
            K = len(np.unique(Y))
        else:
            K = len(np.unique(Y)) - 1
        
        iAnn = Y_metadata
        m_f, m_g = m[:, :K], m[:, K:]
        v_f, v_g = v[:, :K], v[:, K:]
        
        #Convert labels in the codification 1_of_K
        Yhat = np.ones((N, K, R))
        for r in range(R):
            Yhat[:,:,r] = self.one_of_K(Y[:,r,None], K)
            
        
        # E_{q(f_{1,n})...q(f_{K,n})}[log zeta]
        var_exp_log_zeta,_,_ = self.GaussHermiteMC(gh_f, gh_w, m_f, v_f, K)
        
        
        
        # E_{q(g_m^m)}[z_n^m] 
        Eq_g = np.empty((m.shape[0],R))
        for r in range(R):
            alpha_im = gh_f[None, :] * np.sqrt(2. * v_g[:,r:r+1]) + m_g[:,r:r+1]
            zn = self.logisticFunc(alpha_im)
            Eq_g[:,r] = (1/np.sqrt(np.pi))*zn.dot(gh_w[:,None]).flatten()
        # Eq_g = self.KappaFuncEx(m_g, v_g)
        # Eq_g = np.clip(Eq_g, 1e-9, 1. - 1e-9) #numerical stability
        
        # The term \sum_{k=1}^{K}C_{k,n}^r\log(\zeta_{k,n})
        A = np.sum((Yhat*np.reshape(var_exp_log_zeta, (N, K, 1))), axis = 1)

        
        #Variational Expectation ##########################################
        var_exp = np.sum( ( Eq_g*A + np.log(1/K)*(1-Eq_g) )*iAnn, 1).reshape((N,1))
        
        
        return var_exp

    def var_exp_derivatives(self, Y, m, v, GN=None, gh_points=None, Y_metadata=None):
        # Variational Expectations of derivatives
        # gh: Gaussian-Hermite quadrature
        if gh_points is None:
            gh_f, gh_w = self._gh_points()
        else:
            gh_f, gh_w = gh_points
        
        N, R = Y.shape
        Clases = np.unique(Y)
        auxC = np.prod(Clases)
        if auxC > 0:
            K = len(np.unique(Y))
        else:
            K = len(np.unique(Y)) - 1
        
        iAnn = Y_metadata
        m_f, m_g = m[:, :K], m[:, K:]
        v_f, v_g = v[:, :K], v[:, K:]
        
        #Convert labels in the codification 1_of_K
        Yhat = np.ones((N, K, R))
        for r in range(R):
            Yhat[:,:,r] = self.one_of_K(Y[:,r,None], K)
        
            
        # E_{q(g_m^m)}[z_n^m] 
        Eq_g = np.empty((m.shape[0],R))
        for r in range(R):
            alpha_im = gh_f[None, :] * np.sqrt(2. * v_g[:,r:r+1]) + m_g[:,r:r+1]
            zn = self.logisticFunc(alpha_im)
            Eq_g[:,r] = (1/np.sqrt(np.pi))*zn.dot(gh_w[:,None]).flatten()
            
            
        # E_{q(f_{1,n})...q(f_{K,n})}[log zeta]
        var_exp_log_zeta, var_exp_zeta, var_exp_zeta2 = self.GaussHermiteMC(gh_f, gh_w, m_f, v_f, K)
        

        #Derivates #######################################################
        var_exp_dm = np.empty((m.shape[0],R+K))
        var_exp_dv = np.empty((m.shape[0],R+K)) 
        
        # E_{q(f_{1,n})...q(f_{K,n})}[zeta]
        
        
        # Auxxx = np.empty((m.shape[0],K))
        # auxDif = np.empty((m.shape[0],R))
        # for k in range(K):
        #     auxDif = np.matlib.repmat(var_exp_zeta[:,k:k+1],1,R)
        #     Auxxx[:,k] = -0.5*np.sum(Eq_g*(auxDif - auxDif**2)*iAnn, 1)
        

        var_exp_dm[:, :K] = np.sum(np.reshape(Eq_g*iAnn, (N,1,R))* \
                                   (Yhat - np.reshape(var_exp_zeta, (N, K, 1))), axis=2)     
        var_exp_dv[:, :K] = -0.5*np.sum( (np.reshape(Eq_g*iAnn, (N,1,R))* \
                                          (np.reshape(var_exp_zeta, (N, K, 1)) - \
                                           np.reshape(var_exp_zeta2, (N, K, 1)))), axis=2)
        
        
        Const = np.sum((Yhat*np.reshape(var_exp_log_zeta, (N, K, 1))), axis = 1) + np.log(K)
        for r in range(R):
            alpha_im = gh_f[None, :] * np.sqrt(2. * v_g[:,r:r+1]) + m_g[:,r:r+1]
            znm  = self.logisticFunc(alpha_im)
            znm2 = znm**2
            znm3 = znm**3
            Eznm  = (1/np.sqrt(np.pi))*znm.dot(gh_w[:,None])
            Eznm2 = (1/np.sqrt(np.pi))*znm2.dot(gh_w[:,None])
            Eznm3 = (1/np.sqrt(np.pi))*znm3.dot(gh_w[:,None])
            
            
            var_exp_dm[:, r+K] = ((Eznm - Eznm2)*Const[:,r:r+1]*iAnn[:,r:r+1]).flatten()
            var_exp_dv[:, r+K] = (0.5*(2*Eznm3 - 3*Eznm2 + Eznm)*Const[:,r:r+1]*iAnn[:,r:r+1]).flatten()
        
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
        
        # The functions related to the classification scheme 
        N, Nf = m.shape
        mean_pred = []
        var_pred = []
        auxm = m[:,:self.K]
        auxv = v[:,:self.K]
        
        # gh: Gaussian-Hermite quadrature
        gh_f, gh_w = np.polynomial.hermite.hermgauss(5)

        expanded_F_tuples = []
        grid_tuple = [auxm.shape[0]]
        for k in range(self.K):
            grid_tuple.append(gh_f.shape[0])
            expanded_fd_tuple = [1]*(self.K+1)
            expanded_fd_tuple[k+1] = gh_f.shape[0]
            expanded_F_tuples.append(tuple(expanded_fd_tuple))
            
        # mean-variance tuple
        mv_tuple = [1]*(self.K+1)
        mv_tuple[0] = auxm.shape[0]
        mv_tuple = tuple(mv_tuple)
        
        # building, normalizing and reshaping the grids
        F = np.zeros((reduce(lambda x, y: x * y, grid_tuple),self.K))
        for d in range(self.K):
            fd = np.zeros(tuple(grid_tuple))
            fd[:] = np.reshape(gh_f, expanded_F_tuples[d])*np.sqrt(2*np.reshape(auxv[:,d],mv_tuple)) \
                    + np.reshape(auxm[:,d],mv_tuple)
            F[:,d,None] = fd.reshape(reduce(lambda x, y: x * y, grid_tuple), -1, order='C')
            
        # mean
        S_f = self.softmax(F)
        E_S_fk = np.zeros((N,self.K))
        for k in range(self.K):
            S_fk = S_f[:,k].reshape(tuple(grid_tuple))
            E_S_fk1 = S_fk.dot(gh_w) / np.sqrt(np.pi)
            for kl in range(self.K-1):
                E_S_fk1 = E_S_fk1.dot(gh_w) / np.sqrt(np.pi)
            E_S_fk[:,k] = E_S_fk1
        mean_pred.append(E_S_fk)
        
        # variance
        S_f_2 = S_f**2
        V_S_fk = np.zeros((N,self.K))
        for k in range(self.K):
            V_fk = S_f_2[:,k].reshape(tuple(grid_tuple))
            V_S_fk1 = V_fk.dot(gh_w) / np.sqrt(np.pi)
            for kl in range(self.K-1):
                V_S_fk1 = V_S_fk1.dot(gh_w) / np.sqrt(np.pi)
            V_S_fk[:,k] = V_S_fk1 - E_S_fk[:,k]**2
        var_pred.append(V_S_fk)
        
        #The mean and variance for the annotators parameters
        gh_w = gh_w / np.sqrt(np.pi)
        for nf in range(Nf-self.K):
                        
            auxm = m[:,nf+self.K,None]
            auxv = v[:,nf+self.K,None]
            x = gh_f[None, :] * np.sqrt(2. * auxv) + auxm
            
            # The mean function 
            sig_fj_1 = self.logisticFunc(x)
            m_sig_fj = sig_fj_1.dot(gh_w[:,None])
            mean_pred.append(m_sig_fj)
            
            #The variance function
            sig_fj_2 = sig_fj_1**2
            sig_fj_2 = sig_fj_2.dot(gh_w[:,None])
            v_sig_fj = sig_fj_2 - m_sig_fj**2
            var_pred.append(v_sig_fj)
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

    def get_metadata(self, iann):
        _, R = iann.shape
        
        Clases = np.unique(self.Y)
        auxC = np.prod(Clases)
        if auxC > 0:
            K = len(Clases)
        else:
            K = len(Clases) - 1
        
        dim_y = 1
        dim_f = R + K
        dim_p = 1
        return dim_y, dim_f, dim_p

    def ismulti(self):
        # Returns if the distribution is multivariate
        return False
