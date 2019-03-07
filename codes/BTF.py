from . import utils 
from . import BMF

import tensorly as tl
import tensorly.decomposition as td
import tensorly.tenalg as tt

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
    
    dim = len(X.shape)
    list_non_mode = [i for i in range (0, dim)]
    list_non_mode.remove(mode)
    dim_temp = 1
    for i in range (0, len(liste_non_mode)):
        dim_temp *= X.shape[liste_non_mode[i]]
    tenseur_unfold = np.empty([X.shape[mode], dim_temp])
    for i in range (0, X.shape[mode]):
        tenseur_unfold[i, :]=X[:, i, :, :].ravel()
        
    return(tenseur_unfold)

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
    res = tens_temp[0]
    for i in range (1, rank):
        res += temp_temp[i]
    res [res > 1] = 1
    
    return (res)
def create_tensor(size, rank, ):
    ''' create a N-way array from '''
    dim = len(size)
    tens_temp=[]
    sol=[[] for i in range (0,dim)]
    for i in range (0,rang):
        
        v_end_1=np.random.choice([0,1], size=(1,tailles[dim-2]), p=[proportion, 1-proportion])
        v_end_1=v_end_1.astype(float)
        v_end=np.random.choice([0,1], size=(1,tailles[dim-1]), p=[proportion, 1-proportion])
        v_end=v_end.astype(float)
        res_temp = np.outer(v_end_1, v_end)
        sol[dim-1].append(v_end.tolist()[0])
        sol[dim-2].append(v_end_1.tolist()[0])
        
        for j in range (dim-3,-1,-1):
            v_temp=np.random.choice([0,1], size=(1,tailles[j]), p=[proportion, 1-proportion])
            v_temp=v_temp.astype(float)
            sol[j].append(v_temp.tolist()[0])
            res_temp = np.array([[res_temp*i][0] for i in v_temp.ravel()])
        tens_temp.append(res_temp)
    res = tens_temp[0]
    for i in range (1, rang):
        res += tens_temp[i]
    res [res >1] = 1
    return (res, sol)
        
    

