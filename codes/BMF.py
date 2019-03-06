from . import utils


import numpy as np 
from sklearn.decomposition import non_negative_factorization
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

__all__=['c_pnl_pf']
    
def c_pnl_pf(X, rank, n_iter, gamma, lamb, beta, eps, W_ini=False, H_ini=False):
    ''' 
    Factorize a binary matrix into two binary matrices 

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
        psi_W_H = omega_W_H * utils.sigmaf_v(WH, gamma, 0.5)

        mat_sum_H = np.array([sum(H).tolist() for i in range(W.shape[0])])
        mat_sum_W = np.array([sum(W).tolist() for i in range(H.shape[0])])
        sum_W_H = sum(np.dot(W, H.T).ravel())

        H *= (gamma * np.dot((X * omega_W_H).T, W) + 3 * lamb * H**2 + beta * mat_sum_W / (sum_W_H**2)) / (gamma * np.dot(psi_W_H.T, W) + 2 * lamb * H**3 + lamb * H)

        W *= (gamma * np.dot((X * omega_W_H), H) + 3 * lamb * W**2 + beta * mat_sum_H / (sum_W_H**2)) / (gamma * np.dot(psi_W_H.T, H) + 2 * lamb * W**3 + lamb * W)

    H = utils.threshold(H, 0.5)
    W = utils.threshold(W, 0.5)
    return (W,H)

def create_quick_matrix(n_rows, n_columns, n_sources, density, recup=False, opt_print=True):
    '''
    Create a numpy matrix array from sources generate randomly
    (each elements of sources and abundances are a result of a 
    Bernoulli)
    
    
    Paramaters
    ----------
    n_rows : number of rows 
    
    n_columns : number of columns
    
    n_sources : number of sources
    
    density : Bernoulli parameter
    
    recup : if True he get the generated matrix and
    the source and the abundance ones, if False you
    get only the resulted matrix (default = False)
    
    opt_print : if True print the percent of ones 
    in the resulted matrix (default = True)
    
    Examples
    --------
    
    >>> X = create_quick_matrix(50, 50, 3, 0.4)
    >>> X, W, H = create_quick_matrix(500, 100, 8, 0.2, recup = True)
    
    '''
    
    H = np.random.choice([0, 1], size = (n_sources, n_columns), p = [1-density, density])
    W = np.random.choice([0, 1], size = (n_rows, n_sources), p = [1-density, density])
    X = np.dot(W,H)
    X [X>1] = 1
    if opt_print==True:
        print (sum(X.ravel())/(n_rows * n_columns) * 100)
    if recup == False:
        return X
    else: return (X, W, H)


def thresholding(X, W_ini, H_ini):
    ''' Algorithm of thresholding from Binary Matrix Factorization with Applications by Zhang

    '''
    II = np.max(H_ini)
    testh = np.linspace(0, II, int((II - 0) / 0.01))
    ll = np.max(W_ini)
    testw = np.linspace(0, II, int((ll - 0) / 0.01))
    temp = 10**10
    for i in range (len(testh)):
        for j in range (len(testw)):
            newH = signstar(H, testh[i])
            newW = signstat(W, testw[j])
            newtemp = utils.frobenius(X, np.dot(W, H.T))
            if newtemp < temp:
                temp = newtemp
                h = testh[i]
                w = testw[j]
    return (w, h)        

def signstar(a, param):
    ''' 
    Function from Zhang
    '''
    a [a > param] =1
    a [a <= param] = 0
    return a
    
    
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





