import utils


import numpy as np 
from sklearn.decomposition import non_negative_factorization
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def c_pnl_pf(X, rank, n_iter, gamma, lamb, beta, eps, W_ini=False, H_ini=False):
    ''' factorize a binary matrix into two binary matrices 

    Parameters
    ----------
    X : binary numpy matrix arrays
    
    rank : wishes rank of decomposition

    n_iter : maximum of iterations if our stop criteron is not satisfied

    gamma : curvature modifier for sigmoid function

    lamb : binary penalty

    beta : maximal support penalty

    eps : tolerated error for stop criteron

    W_ini : initialised matrix

    H_ini : initialised matrix

    '''


    if (W_ini == False or H_ini == False):
        W_ini, H_ini, thash = non_negative_factorization(X, n_components=rank, solver='mu')

    W, H = utils.normalization(W_ini, H_ini)

    for i in range (n_iter):
        WH = np.dot(W, H.T)
        omega_W_H = utils.calcul_exp_v(WH, gamma, 0.5) * (utils.sigmaf_v(WH, gamma, 0.5)**2)
        psi_W_H = omega_W_H*utils.sigmaf_v(WH, gamma, 0.5)

        mat_sum_H = np.array([sum(H).tolist() for i in range(W.shape[0])])
        mat_sum_W = np.array([sum(W).tolist() for i in range(H.shape[0])])
        sum_W_H = sum(np.dot(W, H.T).ravel())

        H *= (gamma * np.dot((X * omega_W_H).T, W) + 3 * lamb * H**2 + beta * mat_sum_W / (sum_W_H**2)) / (gamma * np.dot(psi_W_H.T, W) + 2 * lamb * H**3 + lamb * H)

        W *= (gamma * np.dot((X * omega_W_H), H) + 3 * lamb * W**2 + beta * mat_sum_H / (sum_W_H**2)) / (gamma * np.dot(psi_W_H.T, H) + 2 * lamb * W**3 + lamb * W)

    H = utils.threshold(H, 0.5)
    W = utils.threshold(W, 0.5)
    return (W,H)

def create_quick_matrix(n,m,k, param, recup=False, opt_print=False):
    proportion=1-param
    H = np.random.choice([0, 1], size = (k,m), p = [proportion, 1-proportion])
    W = np.random.choice([0, 1], size = (n,k), p = [proportion, 1-proportion])
    X = np.dot(W,H)
    X [X>1] = 1
    if opt_print==False:
        print (sum(X.ravel())/(n*m)*100)
    if recup == False:
        return X
    else: return (X, W, H)


def thresholding(X,rank, gamma, w0, h0, n_iter, W_ini=False, H_ini=False):
    ''' Algorithm of thresholding from Binary Matrix Factorization with Applications

    '''

    if (W_ini == False or H_ini == False):
        W_ini, H_ini, thash = non_negative_factorization(X, n_components=rank, solver='mu')
    
    W, H = utils.normalization(W_ini, H_ini)
    alpha_k = 1
    
    for i in range (n_iter):
        omega_W = gamma * utils.calcul_exp_v(W-w0, gamma, 0.5) * (utils.sigmaf_v(W-w0, gamma, 0.5)**2)
        omega_H = gamma * utils.calcul_exp_v(H-h0, gamma, 0.5) * (utils.sigmaf_v(H-h0, gamma, 0.5)**2)
        gk_1 = sum(((np.dot(X, H) - np.dot(np.dot(W, H), H.T)) * omega_W).ravel())
        gk_2 = sum(((np.dot(W.T, X) - np.dot(np.dot(W.T, W), H)) * omega_H).ravel())

        for decrease in range (n_iter_descrease):
            d_k = 1/k^2
            alpha_k = alpha_k - d_k * (sum(X * gamma * utils.calcul_exp_v(W-w0, gamma, 0.5) * (utils.sigmaf_v(W-w0, gamma, 0.5)**2)))

def pf_zhang(X ,rank ,lamb ,nbiter=20, W_ini=False, H_ini=False, eps=10**(-1), esp_nn=10e-8, n_ini=1):
    '''
    Algorithm from Binary Matrix Factorization with Applications
    '''

    if (W_ini == False or H_ini == False):
        W_ini, H_ini, thash = non_negative_factorization(X, n_components=rank, solver='mu')
    
    W, H= normalization(W_ini,H_ini)
    
    for i in range (nbiter):
        H *= (np.dot(X.T, W) + 3 * lamb * H**2) / (np.dot(np.dot(H, W.T), W) + 2 * lamb * H**3 + lamb * H + esp_nn)
        W *= (np.dot(X, H) + 3 * lamb * W**2) / (np.dot(np.dot(W, H.T), H) + 2 * lamb * W**3 + lamb * W + esp_nn)

        if sum(((H**2 - H)**2).ravel()+((W**2 - W)**2).ravel()) < eps:
            W = utils.threshold(W, 0.5)
            H = utils.threshold(H, 0.5)            
            return (W,H)
        lamb = 10 * lamb

    W = utils.threshold(W, 0.5)
    H = utils.threshold(H, 0.5)  
    return (W,H)





