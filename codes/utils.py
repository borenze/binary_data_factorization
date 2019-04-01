#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This program contains utility tools.
"""

import numpy as np
import math
from numpy.linalg import inv

__all__=[]


sqrt = np.vectorize(math.sqrt)

def sigmaf(x, gamma, centering):
    ''' Sigmoid function

    Parameters
    ----------
    x : a scalar 

    gamma : curvature modifier

    centering : the sigmoid function have a center of symmetry which 
    is in centering

    Return
    ------
    y : a scalar

    Example
    -------
    >>> x
    0.4
    >>> gamma
    15
    >>> centering
    0.5
    >>> sigmaf(x, gamma, centering)
    array(0.1824552)

    '''

    return(1/(1+math.exp(-gamma*(x-centering))))


def calcul_exp(x, gamma, centering):
	''' Fonction used in the derivate of sigmaf
	'''
	return(math.exp(-gamma*(x-centering)))

# Here we vectorize these 2 functions to support any type of arrays

sigmaf_v = np.vectorize(sigmaf)
calcul_exp_v = np.vectorize(calcul_exp)

def diag_max(matrix, axe=0):
	''' Funtion which return a diagonal matrix built with the maximum of each row or column

	Parameters
	----------
	matrix : matrix numpy arrays

	axe : 0 to return a diagonal matrix with the maximum of each columns, 1 for rows (default = 0)

	Return
	------
	D : matrix numpy arrays

	Examples
	--------
    >>> X = np.array([[1,2,3],[4,5,6]])
    >>> D = diag_max(X)
    >>> D
    array([[4, 0, 0],
           [0, 5, 0],
           [0, 0, 6]])
    >>> D2 = diag_max(X, axe=1)
    >>> D2
    array([[3, 0],
           [0, 6]])
	'''

	return (np.diag(matrix.max(axis = axe)))

def normalization(W, H):
	''' Normalize matrices W and H 

	Parameters
	----------
	W and H : matrix numpy arrays

	Returns
	-------
	W_norm and H_norm : matrix numpy arrays s.t. X = np.dot(W_norm, H_norm.T)


	'''

	if W.shape[1]!=H.shape[0]:
		H = H.T
		if W.shape[1]!=H.shape[0]:
			return("Matrix dimensions do not agree, please check that X = np.dot(W, H) or X = np.dot(W, H.T)")
	ind = np.where(sum(utils.diag_max(W)) = 0)
	ind2 = np.where(sum(utils.diag_max(H.T)) = 0)
	W[ind[0], ind[0]] = 0.01
	H[ind2[0], ind2[0]] = 0.01
	W_norm = np.dot(np.dot(W, inv(sqrt(diag_max(W, axe = 0)))), sqrt(diag_max(H, axe = 1)))
	H_norm = np.dot(np.dot(inv(sqrt(diag_max(H, axe = 1))), sqrt(diag_max(W, axe = 0))), H)

	return(W_norm, H_norm.T)


def frobenius(matrix1, matrix2):
	''' Calcul square of Frobenius distance between both matrices

	Parameters
	----------
	matrix1 and matrix2 : matrix numpy arrays

	Return
	------
	D_frobe : a scalar

	'''

	D_frobe = sum((matrix1.ravel()-matrix2.ravel())**2)
	return (D_frobe)

def threshold(X, tau):
	''' threshold entries into binary matrices/tensors
    Parameters
    ----------

    X : numpy arrays

    tau : parameter of threshold

	'''
	X [X <= tau] = 0
	X [X > tau] = 1
	
	return X



