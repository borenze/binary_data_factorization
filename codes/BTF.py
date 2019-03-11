from . import utils 
from . import BMF

import tensorly as tl
import tensorly.decomposition as td
import tensorly.tenalg as tt
import numpy as np
import copy
import warnings
import sys
from numpy.linalg import inv
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
    
def create_tensor(size_m, n_sources, density, recup=False):
    ''' create a N-way array from any number of sources generate via Bernoulli
    Parameters
    ----------
    size_m : a list with the number of rows wishes for each matrix 
    
    rank : the number of sources 
    
    density : Bernoulli parameter (between 0 and 1)
    
    recup : if True returns the matrices and the tensor, else returns only 
    the tensor (default = False)
    
    Returns
    -------
    res : a N-way array (N=n_sources)
    
    sol : a list of matrices (if recup==True)
    
    Examples
    --------
    >>> res = create_tensor([50, 100, 140], 4, 0.4)
    >>> res2, sol = create_tensor([10, 15, 20, 25], 3, 0.1, recup=True)
    '''
    dim = len(size_m)
    tens_temp = []
    sol = [[] for i in range (0, dim)]
    for i in range (n_sources):
        
        v_end_1 = np.random.choice([0, 1], size = (1, size_m[dim - 2]), p = [1 - density, density])
        v_end_1 = v_end_1.astype(float)
        v_end=np.random.choice([0,1], size = (1, size_m[dim - 1]), p = [1 - density, density])
        v_end=v_end.astype(float)
        res_temp = np.outer(v_end_1, v_end)
        sol[dim-1].append(v_end.tolist()[0])
        sol[dim-2].append(v_end_1.tolist()[0])
        
        for j in range (dim-3,-1,-1):
            v_temp = np.random.choice([0,1], size = (1, size_m[j]), p = [1 - density, density])
            v_temp = v_temp.astype(float)
            sol[j].append(v_temp.tolist()[0])
            res_temp = np.array([[res_temp * i][0] for i in v_temp.ravel()])
        tens_temp.append(res_temp)
    res = tens_temp[0]
    for i in range (1, n_sources):
        res += tens_temp[i]
    res [res > 1] = 1
    if recup == False:
        return res
    return (res, sol)


def unfolding(X, mode):
    ''' unfold a three-way array
    
    Parameters
    ----------
    X : a three-way array
    
    mode : the axe of unfolding
    
    Return
    ------
    tensor_unfold : a numpy matrix
    
    Examples
    --------
    >>> unfold1 = unfolding(X, 0)
    >>> unfold2 = unfolding(X, 1)
    >>> unfold3 = unfolding(X, 2)
    '''
    
    if mode==0:
        tensor_unfold = np.zeros((X.shape[0], X.shape[1] * X.shape[2]))
        for i in range (X.shape[2]):
            tensor_unfold[:, i * X.shape[1] : (i + 1) * X.shape[1]] = X[:, :, i]
    if mode==1:
        tensor_unfold = np.zeros((X.shape[1], X.shape[0] * X.shape[2]))
        for i in range (X.shape[2]):
            tensor_unfold[:, i * X.shape[0]:(i+1) * X.shape[0]]=X[:, :, i].T
    if mode==2:
        tensor_unfold = np.zeros((X.shape[2], X.shape[0] * X.shape[1]))
        for i in range (X.shape[2]):
            tensor_unfold[i, :] = X[:, :, i].T.ravel()
    
    return(tensor_unfold)




        
def t_pnl_pf(X, rank, n_iter, gamma, lamb, eps, init = False, normalize_opt = False):
    ''' 
    Factorize a n order binary tensor into n binary matrices 

    Parameters
    ----------
    X : binary numpy tensor 
    
    rank : wishes rank of decomposition

    n_iter : maximum of iterations if our stop criteron is not satisfied

    gamma : curvature modifier for sigmoid function

    lamb : binary penalty

    beta : maximal support penalty

    eps : tolerated error for stop criteron

    init : list of initialisation matrices 

    '''
    dim = len(X.shape)

    if (init == False):
        init = td.non_negative_parafac(X, rank=rank)

    if normalize_opt == True:
        for i in range (dim):
            for j in range (rank):
                init[i][:,j] = init[i][:,j]/np.max(init[i][:,j])
                # this option hightly denature our init structure
    unfold_tensor = [unfolding(X, i) for i in range (dim)]               

    for k in range (n_iter):
        for j in range (dim):
            
            list_ind = [i for i in range (dim)] # we create a list of indices for take the right matrices in each Khatri-Rao product
            list_ind.remove(j)
            KR_product = tt.khatri_rao(matrices = [init[i] for i in list_ind[::-1]])
        
            WH = np.dot(init[j], KR_product.T)
            
            omega_W_H = utils.calcul_exp_v(WH, gamma, 0.5) * (utils.sigmaf_v(WH, gamma, 0.5)**2)
            psi_W_H = omega_W_H * utils.sigmaf_v(WH, gamma, 0.5)
      
            
            init[j] *= (gamma * np.dot((unfold_tensor[j] * omega_W_H), KR_product) + 3 * lamb * init[j]**2) / (gamma * np.dot(psi_W_H, KR_product) + 2 * lamb * init[j]**3 + lamb * init[j])
            
    
    for i in range (dim):
        init[i] = utils.threshold(init[i], 0.5)
    
    return (init)


def bt_admm(X, rank, n_iter, n_intern1, n_intern2, alpha, alpha2, gamma, lamb, eps, init = False, normalize_opt = False):
    ''' 
    Factorize a n order binary tensor into n binary matrices 

    Parameters
    ----------
    X : binary numpy tensor 
    
    rank : wishes rank of decomposition

    n_iter : maximum of iterations if our stop criteron is not satisfied

    gamma : curvature modifier for sigmoid function

    lamb : binary penalty

    beta : maximal support penalty

    eps : tolerated error for stop criteron

    init : list of initialisation matrices 

    '''
    dim = len(X.shape)

    if (init == False):
        init = td.non_negative_parafac(X, rank=rank)

    if normalize_opt == True:
        for i in range (dim):
            for j in range (rank):
                init[i][:,j] = init[i][:,j]/np.max(init[i][:,j])
                # this option hightly denature our init structure
    unfold_tensor = [unfolding(X, i) for i in range (dim)]  
    init_barre=copy.deepcopy(init) # We initialize \bar{W}, \bar{H} and \bar{V} 
    tensor_temp=[]
    for j in range(dim):
        tensor_temp.append(init[j] * 0) # We initialize matrices called A, B and C in the reference
        
    for k in range(n_iter):
        for j in range (dim):
            for intern in range (n_intern1):
                list_ind = [i for i in range (dim)]
                list_ind.remove(j)
                KR_product = tt.khatri_rao(matrices=[init[i] for i in liste_ind[::-1]])
                for intern1 in range (n_intern2):
                    WH = np.dot(init_barre[j], KR_product.T)
                    omega_W_H = utils.calcul_exp_v(WH, gamma, 0.5) * (utils.sigmaf_v(WH, gamma, 0.5)**2)
                    psi_W_H = omega_W_H * utils.sigmaf_v(WH, gamma, 0.5)
                    init_barre[j] = init_barre[j] - alpha * (- gamma * (np.dot(depli[j] * omega_W_H, KR_product))\
                                    + gamma * np.dot(psi_W_H, KR_product) - rho * (init[j] - init_barre[j] + tensor_temp[j]))
                for intern2 in range (n_intern2):
                    init[j] = init[j] - alpha2 * (lamb * (init[j] - 3 * init[j]**2 + 2 * init[j]**3) + rho\
                                                 * (init[j] - init_barre[j] + tensor_temp[j]))
                tensor_temp[j] = tensor_temp[j] + init[j] - init_barre[j]
    for i in range (dim):
        for j in range (rank):
            init[i][:,j] = init[i][:,j]/np.max(init[i][:,j])
            
    for i in range (dim):
        init[j] = utils.threshold(init[j], 0.5)
    return init
    




def rebuild_tensor(X):
    ''' giving matrices create tensor
    Parameter
    ---------        
    X : a list of numpy matrices 
       
    Return
    ------
    T : a tensor
        
    Example
    -------
    >>> T = rebuild(X)
    '''
    
    dim = len(X)
    rank = X[0].shape[1]
    tensor_temp = []
    for i in range (rank):
        res_temp = np.outer(X[dim-2][:, i], X[dim-1][:, i])
        for j in range (dim - 3, -1, -1):
            res_temp = np.array([[res_temp * i][0] for i in X[j][:, i]])
        tensor_temp.append(res_temp)
    res = tensor_temp[0]
    for i in range (1, rank):
        res += tensor_temp[i]
    res [res > 1] = 1
    
    return (res)
