from . import utils

import copy
import numpy as np 
from sklearn.decomposition import non_negative_factorization
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

def create_quick_matrix(n_rows, n_columns, n_sources, density, recup = False, opt_print = True):
    '''
    Create a numpy matrix from sources generate randomly
    (each element of sources and abundances are the result of a 
    Bernoulli)
    
    
    Paramaters
    ----------
    n_rows : number of rows 
    
    n_columns : number of columns
    
    n_sources : number of sources
    
    density : Bernoulli parameter
    
    recup : if True you get the generated matrix and
    the source and the abundance ones, if False you
    get only the resulted matrix (default = False)
    
    opt_print : if True print the percent of ones 
    in the resulted matrix (default = True)

    Examples
    --------
    
    >>> from codes import *
    >>> X = BMF.create_quick_matrix(50, 50, 3, 0.4)
    >>> X, W, H = BMF.create_quick_matrix(500, 100, 8, 0.2, recup = True)
    
    Returns
    -------
    
    X : array-like, shape (n_rows, n_columns)
        Result Matrix.
        
    W : array-like, shape (n_rows, n_sources)
        Source Matrix.
    
    H : array-like, shape (n_sources, n_columns)
        Abundance Matrix.
    
    '''
    
    H = np.random.choice([0, 1], size = (n_sources, n_columns), p = [1-density, density])
    W = np.random.choice([0, 1], size = (n_rows, n_sources), p = [1-density, density])
    X = np.dot(W, H)
    X [X > 1] = 1
    if opt_print == True:
        print("The matrix have",round(sum(X.ravel())/(n_rows * n_columns) * 100, 2), "% of ones.")
    if recup == False:
        return X
    else: return (X, W, H)
    

def create_noise_matrix_xor(n_rows, n_columns, n_sources, density, noise, recup_start = True, recup_generated_matrices = False, opt_print = False):
    '''
    Create a numpy matrix from sources generate randomly
    (each elements of sources and abundances are a result of a 
    Bernoulli) with a xor noise
    
    
    Paramaters
    ----------
    n_rows : number of rows 
    
    n_columns : number of columns
    
    n_sources : number of sources
    
    density : Bernoulli parameter
    
    noise : noise parameter (close to 0 for almost empty noise matrix close to 1 for full noise matrix)
    
    recup_genenrated_matrices : if True you get the source and the abundance matrices,
                                if False you don't (default = False)
    
    recup_start : if True you get noiseless matrix if False you don't (default = True)
    
    
    Examples
    --------
    
    >>> from codes import *
    >>> X_noise, X = BMF.create_noise_matrix_xor(50, 50, 3, 0.4, 0.1)
    >>> X_noise, X, W, H = BMF.create_noise_matrix_xor(500, 100, 8, 0.2, noise=0.3, recup_generated_matrices = True,
        opt_print = True)
        
        
    Returns
    -------
    
    X : array-like, shape (n_rows, n_columns)
        Noiseless Matrix.
        
    X_noise : array-like, shape (n_rows, n_columns)
        Noise Matrix.
        
    W : array-like, shape (n_rows, n_sources)
        Source Matrix.
    
    H : array-like, shape (n_sources, n_columns)
        Abundance Matrix.
    '''
    
    if recup_generated_matrices == True:
        X, W, H = create_quick_matrix(n_rows, n_columns, n_sources, density, recup = True, opt_print = False)
    
    else: X = create_quick_matrix(n_rows, n_columns, n_sources, density, opt_print = False)
    
    if opt_print == True:
        print("The noiseless matrix have",round(sum(X.ravel())/(n_rows * n_columns) * 100, 2), "% of ones.")
    noise= np.random.choice([0, 1], size=(X.shape[0],X.shape[1]), p=[1-noise, noise])
    X_noise = X + noise
    X_noise [X_noise > 1] = 0
    
    if recup_generated_matrices == True:
        if recup_start == True:
            return (X_noise, X, W, H)
        return (X_noise, W, H)
    
    if recup_start == True:
        return(X_noise, X)
    
    return X_noise
    
def c_pnl_pf(X, rank, n_iter, lamb, beta, eps, gamma = 5, W_ini = [], H_ini = [], cost_result = False, threshold = True):
    ''' 
    Factorize a binary matrix into two binary matrices 

    Parameters
    ----------
    X : binary numpy matrix arrays
    
    rank : wishes rank of decomposition

    n_iter : maximum of iterations if our stop criteron is not satisfied

    lamb : binary penalty

    beta : maximal support penalty

    eps : tolerated error for stop criteron
    
    gamma : curvature modifier for sigmoid function

    W_ini : initialised matrix

    H_ini : initialised matrix
    
    cost_result : if True return a list with the evolution of the cost function
    
    threshold : if True the returned matrices are threshold to binary 
    
    Examples
    --------
    
    >>> from codes import *
    >>> X_noise, X = BMF.create_noise_matrix_xor(50, 50, 3, 0.4, 0.1)
    >>> W, H = BMF.c_pnl_pf (X_noise, rank = 3, n_iter = 10, lamb = 10, beta = 0, eps = 10**(-8)) 
    

    '''

    # here we define a parameter to forbid devide by zero
    epsilon_non_zero = 10**(-8)
    if (W_ini == [] or H_ini == []):
        W_ini, H_ini, thash = non_negative_factorization(X, n_components=rank, solver='mu')

    W, H = utils.normalization(W_ini, H_ini)
    
    if cost_result == True:
        res_cost = []
        for i in range (n_iter):
            WH = np.dot(W, H.T)
            omega_W_H = utils.calcul_exp_v(WH, gamma, 0.5) * (utils.sigmaf_v(WH, gamma, 0.5)**2)
            psi_W_H = omega_W_H * utils.sigmaf_v(WH, gamma, 0.5)

            mat_sum_H = np.array([sum(H).tolist() for i in range(W.shape[0])])
            mat_sum_W = np.array([sum(W).tolist() for i in range(H.shape[0])])
            sum_W_H = sum(np.dot(W, H.T).ravel())
            res_cost.append(1/2 * utils.frobenius(X, utils.sigmaf_v(WH, gamma, 0.5)) + 1/2 * lamb * utils.frobenius(H, H**2) + 1/2 * lamb * utils.frobenius(W, W**2) + beta * 1 / sum(WH.ravel()))
            
            H *= (gamma * np.dot((X * omega_W_H).T, W) + 3 * lamb * H**2 + beta * mat_sum_W / (sum_W_H**2 + epsilon_non_zero)) / (gamma * np.dot(psi_W_H.T, W) + 2 * lamb * H**3 + lamb * H + epsilon_non_zero)

            W *= (gamma * np.dot((X * omega_W_H), H) + 3 * lamb * W**2 + beta * mat_sum_H / (sum_W_H**2 + epsilon_non_zero)) / (gamma * np.dot(psi_W_H, H) + 2 * lamb * W**3 + lamb * W + epsilon_non_zero)
            res_cost.append(1/2 * utils.frobenius(X, utils.sigmaf_v(WH, gamma, 0.5)) + 1/2 * lamb * utils.frobenius(H, H**2) + 1/2 * lamb * utils.frobenius(W, W**2) + beta * 1 / sum(WH.ravel()))
          
            if (abs(res_cost[i] - res_cost[i-1]) < eps:
                return (W, H, res_cost)
            
        
    if threshold == True:
            H = utils.threshold(H, 0.5)
            W = utils.threshold(W, 0.5)
           
        return (W, H, res_cost)
    
    else:
        res_cost = []
        for i in range (n_iter):
            WH = np.dot(W, H.T)
            omega_W_H = utils.calcul_exp_v(WH, gamma, 0.5) * (utils.sigmaf_v(WH, gamma, 0.5)**2)
            psi_W_H = omega_W_H * utils.sigmaf_v(WH, gamma, 0.5)

            mat_sum_H = np.array([sum(H).tolist() for i in range(W.shape[0])])
            mat_sum_W = np.array([sum(W).tolist() for i in range(H.shape[0])])
            sum_W_H = sum(np.dot(W, H.T).ravel())     
            
            H *= (gamma * np.dot((X * omega_W_H).T, W) + 3 * lamb * H**2 + beta * mat_sum_W / (sum_W_H**2 + epsilon_non_zero)) / (gamma * np.dot(psi_W_H.T, W) + 2 * lamb * H**3 + lamb * H + epsilon_non_zero)

            W *= (gamma * np.dot((X * omega_W_H), H) + 3 * lamb * W**2 + beta * mat_sum_H / (sum_W_H**2 + epsilon_non_zero)) / (gamma * np.dot(psi_W_H, H) + 2 * lamb * W**3 + lamb * W + epsilon_non_zero)
            res_cost.append(1/2 * utils.frobenius(X, utils.sigmaf_v(WH, gamma, 0.5)) + 1/2 * lamb * utils.frobenius(H, H**2) + 1/2 * lamb * utils.frobenius(W, W**2) + beta * 1 / sum(WH.ravel()))
            if (abs(res_cost[i] - res_cost[i-1]) < eps:
                return (W, H)

    
        if threshold == True:
            H = utils.threshold(H, 0.5)
            W = utils.threshold(W, 0.5)
        return (W, H)


def thresholding(X, rank = 0, W_ini = [], H_ini = []):
    ''' Algorithm of thresholding from Binary Matrix Factorization with Applications by Zhang
    '''
    if (rank == 0 and (W_ini == [] or H_ini == [])):
        print(" You have to put initializations or a rank")
        return 
    if (W_ini == [] or H_ini == []):
        W_ini, H_ini, thash = non_negative_factorization(X, n_components=rank, solver='mu')
    W_ini, H_ini = utils.normalization(W_ini, H_ini)
    II = np.max(H_ini)
    testh = np.linspace(0, II, int((II - 0) / 0.01))
    ll = np.max(W_ini)
    testw = np.linspace(0, ll, int((ll - 0) / 0.01))
    temp = 10**10
    for i in range (len(testh)):
        newH = signstar(H_ini, testh[i])
        for j in range (len(testw)):            
            newW = signstar(W_ini, testw[j])
            X_res = np.dot(newW, newH.T)
            X_res [X_res > 1] = 1
            newtemp = utils.frobenius(X, X_res)
            if newtemp < temp:
                temp = newtemp
                h = testh[i]
                w = testw[j]
    return (w, h)        

def signstar(a, param):
    ''' 
    Function from Zhang
    '''
    b = copy.deepcopy(a)
    b = b * 0
    b [a >= param] =1
    return b
    
def pf_zhang(X ,rank ,lamb ,nbiter=10, W_ini=False, H_ini=False, eps=10**(-1), esp_nn=10e-8, n_ini=1):
    '''
    Algorithm from Binary Matrix Factorization with Applications
    '''

    if (W_ini == False or H_ini == False):
        W_ini, H_ini, thash = non_negative_factorization(X, n_components = rank, solver = 'mu')
    
    W, H= utils.normalization(W_ini, H_ini)
    X = X.astype(float)
    W = W.astype(float)
    H = H.astype(float)
    for i in range (nbiter):

        H *= (np.dot(X.T, W) + 3 * lamb * H**2) / (np.dot(np.dot(H, W.T), W) + 2 * lamb * H**3 + lamb * H + esp_nn)
        W *= (np.dot(X, H) + 3 * lamb * W**2) / (np.dot(np.dot(W, H.T), H) + 2 * lamb * W**3 + lamb * W + esp_nn)

        if (sum(((H**2 - H)**2).ravel())+sum(((W**2 - W)**2).ravel())) < eps:
            W = utils.threshold(W, 0.5)
            H = utils.threshold(H, 0.5)            
            return (W,H)

    W = utils.threshold(W, 0.5)
    H = utils.threshold(H, 0.5)  
    return (W, H)


def pf_threshold (X, rank, n_iter, lamb, beta, th=1, W_ini = [], H_ini = []):
    epsilon_non_zero = 10**(-8)
    if (W_ini == [] or H_ini == []):
        W_ini, H_ini, thash = non_negative_factorization(X, n_components=rank, solver='mu')

    W, H = utils.normalization(W_ini, H_ini)
    for i in range (n_iter):
        WH = np.dot(W, H.T)
        gradient = 0.5 / (th - 0.5)
        D_W_H = WH * 0
        D_W_H [WH < th] = gradient
        H *= (np.dot((X * D_W_H).T, W) + 3 * lamb * H**2) / (np.dot((WH * D_W_H).T, W) + 2 * lamb * H**3 + lamb * H + epsilon_non_zero)
        W *= (np.dot((X * D_W_H), H) + 3 * lamb * W**2) / (np.dot(WH * D_W_H, H) + 2 * lamb * W**3 + lamb * W + epsilon_non_zero)
    H = utils.threshold(H, 0.5)
    W = utils.threshold(W, 0.5)
    return (W, H)







