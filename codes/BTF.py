from . import utils 
from . import BMF

import tensorly as tl
import tensorly.decomposition as td
import tensorly.tenalg as tt

def depliement(X, mode):
    dim = len(X.shape)
    liste_non_mode = [i for i in range (0,dim)]
    liste_non_mode.remove(mode)
    dim_temp = 1
    for i in range (0, len(liste_non_mode)):
        dim_temp *= X.shape[liste_non_mode[i]]
    tenseur_deplie = np.empty([X.shape[mode], dim_temp])
    for i in range (0, X.shape[mode]):
        tenseur_deplie[i,:]=X[:,i,:,:].ravel()

    return(tenseur_deplie)
