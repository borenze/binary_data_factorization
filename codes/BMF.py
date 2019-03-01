import numpy as np
from sklearn.decomposition import NMF
import time
import matplotlib.pyplot as plt
import statistics
from statistics import mean
import os
from sklearn.decomposition import NMF, non_negative_factorization
import csv
import math
import multiprocessing
import sys
import warnings
from numpy.linalg import inv
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from numpy.linalg import inv
import random


##############fonctions utiles pour pnl_pf
sqrt = np.vectorize(math.sqrt)

n_cores=multiprocessing.cpu_count()
def sigmaf(x, lamb, c):
    return (1/(1+math.exp(-lamb*(x-c))))
def exp(x, lamb):
    return math.exp(-lamb*(x-0.5))
def exp2(x, lamb, c):
    return math.exp(-lamb*(x-c))
exp2 = np.vectorize(exp2)
exp = np.vectorize(exp)
sigmaf = np.vectorize(sigmaf)

def diag_max(mat):
    temp = mat.max(axis=0)
    return (np.diag(temp))


def normalization(W,H):
    H = H.T
    D_H = diag_max(H)
    D_W = diag_max(W)
    vec1 = np.where (np.diag(D_H)==0)
    vec2 = np.where (np.diag(D_W)==0)
    D_H[vec1, vec1]=0.1
    D_W[vec2, vec2]=0.1
    D_H_1_2 = sqrt(D_H)
    D_W_1_2 = sqrt(D_W)
    inver_D_W = inv(D_W)
    inver_D_W_1_2 = sqrt(inver_D_W)
    inver_D_H = inv(D_H)
    inver_D_H_1_2 = sqrt(inver_D_H)

    W_norm = np.dot(np.dot(W, inver_D_W_1_2), D_H_1_2)
    H_norm = np.dot(H, np.dot(inver_D_H_1_2, D_W_1_2))
    return (W_norm, H_norm)
#######################
#######################
#######################
def frobenius(X,X1):
    a=X.ravel()-X1.ravel()
    a=a**2
    return(sum(a))

def NMF(liste):
    X=liste[0]
    k=liste[1]
    nbiter=liste[2]
    solver_p=liste[3]
    beta_loss_p=liste[4]
    tol_p=liste[5]
    W, H, n_iter = non_negative_factorization(X, n_components=k, max_iter = nbiter, solver = solver_p, beta_loss=beta_loss_p, tol=tol_p)
    return W,H

def NMF_frob(liste):
    X=liste[0]
    k=liste[1]
    nbiter=liste[2]
    solver_p=liste[3]
    beta_loss_p=liste[4]
    tol_p=liste[5]
    i=liste[6]
    W, H, n_iter = non_negative_factorization(X, n_components=k, max_iter = nbiter, solver = solver_p, beta_loss=beta_loss_p, tol=tol_p,random_state=i)
    frob_dist=frobenius(X,np.dot(W,H))
    return W, H, frob_dist

def multiple_nmf(X, r,n_ini, nbiter=200, solver='mu', beta_loss=2, tol=10e-8, nb_cores=n_cores, random=5):
    inputs = [[X,r,nbiter,solver,beta_loss,tol,random*(i+1)] for i in range (0,n_ini)]
    pool = multiprocessing.Pool(processes=nb_cores)
    pool_outputs = pool.map(NMF_frob, inputs)
    pool.close() 
    pool.join()
    frobe_dist=[result[2] for result in pool_outputs]
    ind = frobe_dist.index(min(frobe_dist))
    return pool_outputs[ind][0], pool_outputs[ind][1]

def multiple_nmf_naif(X, r,n_ini, nbiter=200, solver_p='mu', beta_loss_p=2, tol_p=10e-8):
    err = 10e8
    for i in range (0,n_ini):
        W, H, n_iter = non_negative_factorization(X, n_components=r, max_iter = nbiter, solver = solver_p, beta_loss=beta_loss_p, tol=tol_p)
        if (frobenius(X,np.dot(W,H)/(X.shape[0]*X.shape[1]))< err):
            err = frobenius(X,np.dot(W,H))
            H_keep = H
            W_keep = W

    return W_keep, H_keep










################################
def pnl_pf(X, k, gamma, nbiter, lamb, eps, n_ini=16, max_iter_nmf=200, tol_nmf=10e-4, nb_cores=n_cores, multi_process=True, flag_opti=False,W_ini=[], H_ini=[]):
    n=X.shape[0]
    m=X.shape[1]
    if W_ini==[] and H_ini==[]:
        if multi_process==True:
            W_ini, H_ini = multiple_nmf(X, k, n_ini, max_iter_nmf, solver = 'mu', beta_loss=2, tol=tol_nmf, nb_cores=nb_cores, random=random.randint(1,1800))
        else:
            W_ini, H_ini = multiple_nmf_naif(X, k, n_ini, max_iter_nmf, solver_p = 'mu', beta_loss_p=2, tol_p=tol_nmf)
    W, H= normalization(W_ini,H_ini)
    for i in range (1, (nbiter+1)):
        ewh = sigmaf(np.dot(W,H.T), gamma, 0.5)
        ewh1 = exp(np.dot(W,H.T), gamma)

        #mise à jour de H
        num_H= gamma *np.dot((X*ewh1*(ewh**2)).T,W) + 3 * lamb*(H**2)
        den_H= gamma *np.dot((ewh1*(ewh**3)).T,W) + 2*lamb *(H**3)+lamb*H
        H *= num_H/(den_H+10e-8)

        #mise à jour de W
        num_W= gamma *np.dot((X*ewh1*(ewh**2)),H) + 3 * lamb*(W**2)
        den_W= gamma *np.dot((ewh1*(ewh**3)),H) + 2*lamb *(W**3)+lamb*W
        W *= num_W/(den_W+10e-8)

        if flag_opti==False:
            W,H = normalization(W,H.T)
        if i % 10 == 0:
            if sum(abs((X-sigmaf(np.dot(W,H.T), gamma, 0.5)).ravel()))/(n*m) < eps:
                break
            
    H = sigmaf(H, 200, 0.5)
    W = sigmaf(W, 200, 0.5)
    H [H<=0.5]=0
    H [H >0.5]=1
    W [W<=0.5]=0
    W [W >0.5]=1
    return (W,H)



def c_pnl_pf(X, k, gamma, nbiter, lamb, beta, eps, n_ini=16, max_iter_nmf=200, tol_nmf=10e-4, nb_cores=n_cores, multi_process=True, flag_opti=False):
    n=X.shape[0]
    m=X.shape[1]
    if multi_process==True:
        W_ini, H_ini = multiple_nmf(X, k, n_ini, max_iter_nmf, solver = 'mu', beta_loss=2, tol=tol_nmf, nb_cores=nb_cores, random=random.randint(1,1800))
    else:
        W_ini, H_ini = multiple_nmf_naif(X, k, n_ini, max_iter_nmf, solver_p = 'mu', beta_loss_p=2, tol_p=tol_nmf)

    W, H= normalization(W_ini,H_ini)
    
    for i in range (1, (nbiter+1)):
        ewh = sigmaf(np.dot(W,H.T), gamma, 0.5)
        ewh1 = exp(np.dot(W,H.T), gamma)
        sum_W_H = sum(np.dot(W,H.T).ravel())
        mat_sum_h=np.array([sum(H).tolist() for i in range(W.shape[0])])
       
        mat_sum_w=np.array([sum(W).tolist() for i in range(H.shape[0])])
        #mise à jour de H
        num_H= gamma *np.dot((X*ewh1*(ewh**2)).T,W) + 3 * lamb*(H**2) + beta*mat_sum_w/(sum_W_H**2+eps)
        den_H= gamma *np.dot((ewh1*(ewh**3)).T,W) + 2*lamb *(H**3)+lamb*H
        H *= num_H/(den_H+10e-8)

        #mise à jour de W
        num_W= gamma *np.dot((X*ewh1*(ewh**2)),H) + 3 * lamb*(W**2) + beta*mat_sum_h/(sum_W_H**2+eps)
        den_W= gamma *np.dot((ewh1*(ewh**3)),H) + 2*lamb *(W**3)+lamb*W
        W *= num_W/(den_W+10e-8)
        
        if i % 10 == 0:
            if sum(abs((X-sigmaf(np.dot(W,H.T), 500, 0.5)).ravel()))/(n*m) < eps:
                break
            
    H = sigmaf(H, 200, 0.5)
    W = sigmaf(W, 200, 0.5)
    H [H<=0.5]=0
    H [H >0.5]=1
    W [W<=0.5]=0
    W [W >0.5]=1
    return (W,H)


################optimisation:
def optimisation_rang(X, gamma, nbiter, lamb, eps, n_ini=16, max_rang=20):
    err = []
    for i in range (2, max_rang):
        W, H = pnl_pf(X, i, gamma, nbiter, lamb, eps)
        err.append(frobenius(X,np.dot(W,H.T))/(X.shape[0]*X.shape[1]))
    return (err.index(min(err))+2)
                
def optimisation_random(X, nbiter, eps, iter_nmf, n_ini=16, nb_random=50, k_random=False):
    err=[]
    keep=[]
    k=k_random
    for i in range(0, nb_random):
        gamma = 80 * np.random.random_sample()
        lamb = (900 - 5) * np.random.random_sample() +5
        if k_random==False:
            k = int((25 - 2) * np.random.random_sample() + 2)
        keep.append([k,gamma, lamb])
        W, H = pnl_pf(X, k, gamma, nbiter, lamb, eps, max_iter_nmf=iter_nmf, flag_opti=True)
        err.append(frobenius(X,np.dot(W,H.T))/(X.shape[0]*X.shape[1]))
    ind = err.index(min(err))
    return keep[ind]

def optimisation_random_p(liste):
    X=liste[0]
    nbiter=liste[1]
    eps=liste[2]
    iter_nmf=liste[3]
    n_ini=16
    nb_random=1
    k_random=True
    err=[]
    keep=[]
    k=k_random
    for i in range(0, nb_random):
        gamma = 80 * np.random.random_sample()
        lamb = (900 - 5) * np.random.random_sample() +5
        if k_random==False:
            k = int((25 - 2) * np.random.random_sample() + 2)
        keep.append([k,gamma, lamb])
        W, H = pnl_pf(X, k, gamma, nbiter, lamb, eps, max_iter_nmf=iter_nmf, flag_opti=False)
        err.append(frobenius(X,np.dot(W,H.T))/(X.shape[0]*X.shape[1]))
    ind = err.index(min(err))
    return keep[ind]

def parall_optimisation_random(X, nbiter, eps, iter_nmf, n_ini=16, nb_random=50, k_random=False):
    inputs = [[X,nbiter,eps,iter_nmf] for i in range (0,nb_random)]
    pool = multiprocessing.Pool(processes=15)
    pool_outputs = pool.map(optimisation_random_p, inputs)
    pool.close() 
    pool.join()
    res=[result for result in pool_outputs]
    return res
    
################
def create_quick_matrix(n,m,k, param, recup=False, opt_print=False):
    proportion=1-param
    H = np.random.choice([0, 1], size=(k,m), p=[proportion, 1-proportion])
    W = np.random.choice([0, 1], size=(n,k), p=[proportion, 1-proportion])
    X = np.dot(W,H)
    X [X>1] = 1
    if opt_print==False:
        print (sum(X.ravel())/(n*m)*100)
    if recup == False:
        return X
    else: return X, W, H







#####################
def opti_pnl_pf(X, k, gamma, nbiter, lamb, eps, n_ini=16, max_iter_nmf=200, tol_nmf=10e-4, nb_cores=n_cores, multi_process=True):
    n=X.shape[0]
    m=X.shape[1]
    if multi_process==True:
        W_ini, H_ini = multiple_nmf(X, k, n_ini, max_iter_nmf, solver = 'mu', beta_loss=2, tol=tol_nmf, nb_cores=nb_cores, random=random.randint(1,1800))
    else:
        W_ini, H_ini = multiple_nmf_naif(X, k, n_ini, max_iter_nmf, solver_p = 'mu', beta_loss_p=2, tol_p=tol_nmf)
    W, H= normalization(W_ini,H_ini)
    for i in range (1, (nbiter+1)):
        
        ewh = sigmaf(np.dot(W,H.T), gamma, 0.5)
        ewh1 = exp(np.dot(W,H.T), gamma)
        H_2=H**2
        ewh_2=ewh**2
        ewh_3=ewh_2*ewh
        #mise à jour de H
        num_H=gamma *np.dot((X*ewh1*(ewh_2)).T,W) + 3 * lamb*(H_2)
        den_H=gamma *np.dot((ewh1*(ewh_3)).T,W) + 2*lamb *(H_2*H)+lamb*H
        H *= num_H/(den_H+10e-8)
        del H_2
        W_2=W**2
        #mise à jour de W
        num_W= gamma *np.dot((X*ewh1*(ewh_2)),H) + 3 * lamb*(W_2)
        den_W= gamma *np.dot((ewh1*(ewh_3)),H) + 2*lamb *(W_2*W)+lamb*W
        W *= num_W/(den_W+10e-8)

        if i % 10 == 0:
            H1 = sigmaf(H, 200, 0.5)
            W1 = sigmaf(W, 200, 0.5)
            H1 [H1<=0.5]=0
            H1 [H1 >0.5]=1
            W1 [W1<=0.5]=0
            W1 [W1 >0.5]=1
            if sum(abs((X-np.dot(W,H.T)).ravel()))/(n*m) < eps:
                print('condition de convergence atteinte')
                break
            del W1
            del H1
    H [H<=0.5]=0
    H [H >0.5]=1
    W [W<=0.5]=0
    W [W >0.5]=1
    return (W,H)



def opti_c_pnl_pf(X, k, gamma, nbiter, lamb, eps, beta, n_ini=16, max_iter_nmf=200, tol_nmf=10e-4, nb_cores=n_cores, multi_process=True, W_ini=[], H_ini=[],print_opt=False):
    n=X.shape[0]
    m=X.shape[1]
    if W_ini==[]:
        if multi_process==True:
            W_ini, H_ini = multiple_nmf(X, k, n_ini, max_iter_nmf, solver = 'mu', beta_loss=2, tol=tol_nmf, nb_cores=nb_cores, random=random.randint(1,1800))
        else:
            W_ini, H_ini = multiple_nmf_naif(X, k, n_ini, max_iter_nmf, solver_p = 'mu', beta_loss_p=2, tol_p=tol_nmf)
    W, H= normalization(W_ini,H_ini)
    if print_opt==True:
        res_fc=[]
        res_dfc=[]
    for i in range (1, (nbiter+1)):
        
        ewh = sigmaf(np.dot(W,H.T), gamma, 0.5)
        ewh1 = exp(np.dot(W,H.T), gamma)
        if print_opt==True:
            res_fc.append([sum(((X-ewh)**2).ravel()),lamb/2*(sum(((W-W**2)**2).ravel())+sum(((H-H**2)**2).ravel()))])
            #print(sum(((X-ewh)**2).ravel()),lamb/2*(sum(((W-W**2)**2).ravel())+sum(((H-H**2)**2).ravel())))
        sum_W_H=sum(np.dot(W,H.T).ravel())
        H_2=H**2
        ewh_2=ewh**2
        ewh_3=ewh_2*ewh
        mat_sum_h=np.array([sum(H).tolist() for i in range(W.shape[0])])
        mat_sum_w=np.array([sum(W).tolist() for i in range(H.shape[0])])
        if print_opt==True:
            res_dfc.append([])
        #mise à jour de H
        num_H=gamma *np.dot((X*ewh1*(ewh_2)).T,W) + 3 * lamb*(H_2)+beta*mat_sum_w/(sum_W_H**2+eps)
        den_H=gamma *np.dot((ewh1*(ewh_3)).T,W) + 2*lamb *(H_2*H)+lamb*H
        H *= num_H/(den_H+10e-8)
        del H_2
        W_2=W**2
        #mise à jour de W
        num_W= gamma *np.dot((X*ewh1*(ewh_2)),H) + 3 * lamb*(W_2)+beta*mat_sum_h/(sum_W_H**2+eps)
        den_W= gamma *np.dot((ewh1*(ewh_3)),H) + 2*lamb *(W_2*W)+lamb*W
        W *= num_W/(den_W+10e-8)

        if i % 1 == 0:
            #lamb=lamb-10
            lamb=max(lamb,10)
            H1 = sigmaf(H, 200, 0.5)
            W1 = sigmaf(W, 200, 0.5)
            H1 [H1<=0.5]=0
            H1 [H1 >0.5]=1
            W1 [W1<=0.5]=0
            W1 [W1 >0.5]=1
            X_res=np.dot(W1,H1.T)
            X_res[X_res>1]=1
            #print(sum(abs((X-X_res)).ravel())/(n*m))
            if sum(abs((X-X_res)).ravel())/(n*m) < eps:
                print('condition de convergence atteinte en', i, 'itérations')
                break
            del W1
            del H1
        #lamb=10*lamb
    H [H<=0.5]=0
    H [H >0.5]=1
    W [W<=0.5]=0
    W [W >0.5]=1
    if print_opt==True:
        return(W,H,res_fc)
    return (W,H)



def pf_zhang(X,k,lamb,nbiter=20,W_ini=[],H_ini=[],eps=10**(-1),esp_nn=10e-8,n_ini=1):
    if W_ini==[]:
        W_ini, H_ini = multiple_nmf_naif(X, k, n_ini, 200, solver_p = 'mu', beta_loss_p=2, tol_p=10e-4)
    W, H= normalization(W_ini,H_ini)
    
    for i in range (nbiter):
        #updateH
        H*=(np.dot(X.T,W)+3*lamb*H**2)/(np.dot(np.dot(H,W.T),W)+2*lamb*H**3+lamb*H+esp_nn)
        W*=(np.dot(X,H)+3*lamb*W**2)/(np.dot(np.dot(W,H.T),H)+2*lamb*W**3+lamb*W+esp_nn)
        if sum(((H**2-H)**2).ravel()+((W**2-W)**2).ravel())<eps:
            print('out en',i, 'itérations')
            W[W>0.5]=1
            W[W<0.5]=0
            H[H<0.5]=0
            H[H>0.5]=1
            return (W,H)
        lamb=10*lamb

    W[W>0.5]=1
    W[W<0.5]=0
    H[H<0.5]=0
    H[H>0.5]=1   
    return (W,H)





def thresholding(X,k,w0,h0,nbiter=20,W_ini=[],H_ini=[],eps=10**(-4),gamma=100):
    if W_ini==[]:
        W_ini, H_ini = multiple_nmf_naif(X, k, 1, 200, solver_p = 'mu', beta_loss_p=2, tol_p=10e-4)
    W, H= normalization(W_ini,H_ini)
    for i in range (nbiter):
        mat_coef1=gamma*(exp2(simgaf(W,gamma,0.5),gamma,w0))
        mat_1=np.dot(X,sigmaf(H,gamma,0.5))-np.dot(np.dot(sigmaf(W,gamma,0.5),sigmaf(H.T,gamma,0.5)),sigmaf(H,gamma,0.5))
        g1=sum((mat_1*mat_coef1).ravel())
        #mat_coef2=
        

