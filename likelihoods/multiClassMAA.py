# Copyright (c) 2020 Julian Gil-Gonzalez
# Universidad Tecnologica de Pereira and University of Sheffield

import sys
import numpy as np
from GPy.likelihoods import link_functions
from GPy.likelihoods import Likelihood
from GPy.util.misc import safe_exp, safe_square
from GPy.util.univariate_Gaussian import std_norm_pdf, std_norm_cdf
from functools import reduce # only in Python 3
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score, accuracy_score
from numpy.matlib import repmat


class MultiClassMAA(Likelihood):
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

        super(MultiClassMAA, self).__init__(gp_link, name='MultiClassMAA')

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
    
    def MontecarloMC(self, m, v, MC_S=1000):
        N, J = m.shape
        F = np.zeros((N,MC_S,J))
        for j in range(J):
            F[:,:,j] = np.dot(np.linalg.cholesky(np.diag(v[:,j])+1e-6*np.eye(N)),
                              np.random.randn(N,MC_S)) + m[:,j:j+1]
        return F

    def var_exp(self, Y, m, v, GN=None, gh_points=None, Y_metadata=None):
        # Variational Expectation
        
        N, R = Y.shape
        Clases = np.unique(Y)
        auxC = np.prod(Clases)
        if auxC > 0:
            K = len(np.unique(Y))
        else:
            K = len(np.unique(Y)) - 1
        #Convert labels in the codification 1_of_K
        Yhat = np.ones((N, K, R))
        for r in range(R):
            Yhat[:,:,r] = self.one_of_K(Y[:,r,None], K)
        iAnn = Y_metadata
        N, J = m.shape
        MC_S=10000
        var_exp_C = np.zeros((N,MC_S))
        F = self.MontecarloMC(m, v, MC_S)
        # Montecarlo Simulations
        for i in range(MC_S):
            f_k, f_lr =F[:,i,:K], F[:,i,K:]
            zeta_kn = self.softmax(f_k)
            pi_rn = self.logisticFunc(f_lr)
            prod_k_zeta = np.sum(Yhat*np.reshape(zeta_kn, (N, K, 1)),axis=1)
            var_exp_C[:,i] = np.sum(np.log((pi_rn*prod_k_zeta + (1-pi_rn)/K))*iAnn,axis=1)
        var_exp = np.mean(var_exp_C, axis=1)
        return var_exp

    def var_exp_derivatives(self, Y, m, v, GN=None, gh_points=None, Y_metadata=None):
        # Variational Expectations of derivatives
        N, R = Y.shape
        Clases = np.unique(Y)
        auxC = np.prod(Clases)
        if auxC > 0:
            K = len(np.unique(Y))
        else:
            K = len(np.unique(Y)) - 1
        #Convert labels in the codification 1_of_K
        Yhat = np.ones((N, K, R))
        for r in range(R):
            Yhat[:,:,r] = self.one_of_K(Y[:,r,None], K)
        iAnn = Y_metadata
        N, J = m.shape
        MC_S=10000
        var_exp_dm_C, var_exp_dv_C = np.zeros((N,MC_S,J)), np.zeros((N,MC_S,J))
        F = self.MontecarloMC(m, v, MC_S)
        # Montecarlo Simulations
        for i in range(MC_S):
            f_k, f_lr =F[:,i,:K], F[:,i,K:]
            zeta_kn = self.softmax(f_k)
            pi_rn = self.logisticFunc(f_lr)
            prod_k_zeta = np.sum((Yhat*np.reshape(zeta_kn, (N, K, 1))),axis=1)
            alpha_nr = 1/(pi_rn*prod_k_zeta + (1-pi_rn)/K)
            alpha_nr[alpha_nr < 1e-6] = 1e-6 #numerical stability
            # alpha_nr1 = np.zeros((N,R))
            # for n in range(N):
            #     for r in range(R):
            #         alpha_nr1[n,r] = 1/(pi_rn[n,r]*prod_k_zeta[n,r] + (1-pi_rn[n,r])/K)
            # d_dfk1 = np.zeros((N,K))
            # for n in range(N):
            #     for k in range(K):
            #         aux = 0
            #         for r in range(R):
            #             aux += alpha_nr[n,r]*pi_rn[n,r]*zeta_kn[n,k]*(Yhat[n,k,r]-prod_k_zeta[n,r])
            #         d_dfk1[n,k] = aux
                    

            d_dfk = np.reshape(zeta_kn, (N, K, 1))*np.reshape(pi_rn*alpha_nr, (N, 1, R))*(Yhat 
                                                                                    - np.reshape(prod_k_zeta, (N, 1, R)))
            d_dfk = d_dfk*np.reshape(iAnn, (N,1,R))
            var_exp_dm_C[:,i,:K] = np.sum(d_dfk,axis=2)
            
            
            # d_dflr1 = np.zeros((N,R))
            # for n in range(N):
            #     for r in range(R):
            #         d_dflr1[n,r] = alpha_nr[n,r]*(pi_rn[n,r]-pi_rn[n,r]**2)*(prod_k_zeta[n,r]-1/K)
                    
      
            d_dflr = alpha_nr*(pi_rn-pi_rn**2)*(prod_k_zeta-(1/K))*iAnn
            var_exp_dm_C[:,i,K:] = d_dflr
            
            alpha_nr_2 = alpha_nr**2
            alpha_nr_2[alpha_nr_2 < 1e-6] = 1e-6 #numerical stability
            d_alpha_fk = -np.reshape(pi_rn, (N, 1, R))*(Yhat - 
                        np.reshape(prod_k_zeta, (N, 1, R)))*np.reshape(zeta_kn, (N, K, 1))/np.reshape(alpha_nr_2, (N, 1, R))
            # d_alpha_fk1 = np.zeros((N,K,R))
            # for n in range(N):
            #     for k in range(K):
            #         for r in range(R):
            #             d_alpha_fk1[n,k,r] = -pi_rn[n,r]*zeta_kn[n,k]*(Yhat[n,k,r]-prod_k_zeta[n,r])/alpha_nr[n,r]**2
                        
            
            d2_dfk2_A = np.reshape(alpha_nr, (N, 1, R))*np.reshape(zeta_kn-
                                 zeta_kn**2, (N, K, 1)) + d_alpha_fk*np.reshape(zeta_kn, (N, K, 1))
            # d2_dfk2_A1 = np.zeros((N,K,R))
            # for n in range(N):
            #     for k in range(K):
            #         for r in range(R):
            #             d2_dfk2_A1[n,k,r] = (zeta_kn[n,k] - zeta_kn[n,k]**2)*alpha_nr[n,r] + d_alpha_fk[n,k,r]*zeta_kn[n,k]
                        
                        
            d2_dfk2_B = (d_alpha_fk*np.reshape(zeta_kn, (N, K, 1))*np.reshape(prod_k_zeta, (N, 1, R))+#
                         np.reshape(zeta_kn-zeta_kn**2, (N, K, 1))*np.reshape(alpha_nr, (N, 1, R))*
                         np.reshape(prod_k_zeta, (N, 1, R))+#
                         np.reshape(zeta_kn**2, (N, K, 1))*np.reshape(alpha_nr, (N, 1, R))*
                         (Yhat- np.reshape(prod_k_zeta, (N, 1, R))))#
            # d2_dfk2_B1 = np.zeros((N,K,R))
            # for n in range(N):
            #     for k in range(K):
            #         for r in range(R):
            #             d2_dfk2_B1[n,k,r] = (d_alpha_fk[n,k,r]*zeta_kn[n,k]*prod_k_zeta[n,r]+
            #                                  (zeta_kn[n,k] - zeta_kn[n,k]**2)*alpha_nr[n,r]*prod_k_zeta[n,r]+
            #                                  zeta_kn[n,k]**2*alpha_nr[n,r]*(Yhat[n,k,r]-prod_k_zeta[n,r]))
            
            # d2_dfk21 = np.zeros((N,K))
            # for n in range(N):
            #     for k in range(K):
            #         aux = 0
            #         for r in range(R):
            #             aux += pi_rn[n,r]*(Yhat[n,k,r]*d2_dfk2_A[n,k,r]-
            #                                d2_dfk2_B[n,k,r])
            #         d2_dfk21[n,k] = aux
                        
            d2_dfk2 = (np.reshape(pi_rn, (N, 1, R))*(Yhat*d2_dfk2_A-
                       d2_dfk2_B))
            d2_dfk2 = d2_dfk2*np.reshape(iAnn, (N,1,R))
            var_exp_dv_C[:,i,:K] = np.sum(d2_dfk2,axis=2)
            d_alpha_flr = -(pi_rn-pi_rn**2)*(prod_k_zeta-(1/K))/alpha_nr_2
            # d_alpha_flr1 = np.zeros((N,R))
            # for n in range(N):
            #     for r in range(R): 
            #         d_alpha_flr1[n,r] = -(pi_rn[n,r]-pi_rn[n,r]**2)*(prod_k_zeta[n,r]-1/K)/alpha_nr[n,r]**2
            d2_dflr2 = (d_alpha_flr*(pi_rn-pi_rn**2) +
                        alpha_nr*(2*pi_rn**3-3*pi_rn**2+pi_rn))*(prod_k_zeta-(1/K))*iAnn   
            var_exp_dv_C[:,i,K:] = d2_dflr2
            # d2_dflr21 = np.zeros((N,R))
            # for n in range(N):
            #     for r in range(R): 
            #         d2_dflr21[n,r] = (d_alpha_flr[n,r]*(pi_rn[n,r]-pi_rn[n,r]**2)+
            #                           (2*pi_rn[n,r]**3 -3*pi_rn[n,r]**2+pi_rn[n,r])*alpha_nr[n,r])*(prod_k_zeta[n,r]-1/K)
            
                    
        var_exp_dm = np.mean(var_exp_dm_C,axis=1)
        var_exp_dv = 0.5*np.mean(var_exp_dv_C,axis=1)
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
