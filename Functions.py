#!/usr/bin/env python
# coding: utf-8

# In[1]:


import struct
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot, pylab  


# In[2]:


import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom


# In[3]:


# %load gf2elim.py
import numba

@numba.jit(nopython=True, parallel=True) #parallel speeds up computation only over very large matrices
# M is a mxn matrix binary matrix 
# all elements in M should be uint8 
def gf2elim(M):

    m,n = M.shape

    i=0
    j=0
    h=0
    while i < m and j < n:
        # find value and index of largest element in remainder of column j
        k = np.argmax(M[i:, j]) +i

        # swap rows
        h+=1
        #print(k)
        #M[[k, i]] = M[[i, k]] this doesn't work with numba
        temp = np.copy(M[k])
        M[k] = M[i]
        M[i] = temp


        aijn = M[i, j:]

        col = np.copy(M[:, j]) #make a copy otherwise M will be directly affected

        col[i] = 0 #avoid xoring pivot row with itself

        flip = np.outer(col, aijn)

        M[:, j:] = M[:, j:] ^ flip

        i += 1
        j +=1

    return M


# In[4]:


def Linear_Independence(G,k,n):
    
    A = [[0 for x in range(k)] for y in range(n)]
    indice_indep = [0] * k
    indice_dep = [0] * (n-k)
    
    i = 0
    p = 0
    a = 0
    h = 0
    l = 0
    
    
    while (i != k):
   
        A[i] = G[i+p]
        
        B = gf2elim(np.transpose(A[:(i+1)]))
        for j in range(k):
                
            if(np.linalg.norm(B[j]) != 0):
                    
                a += 1

            else: 
                pass
                
        if(a != i+1): #LD
            indice_dep[p] = i
            p += 1
            a = 0
        else: #LI
            indice_indep[i] = i + p
            i += 1
            a = 0
    mylist = list(range(127))
    j = 0            
    for item in mylist:
        if item in indice_indep:
            pass
        else:
            indice_dep[j] = item
            j += 1   
#    for i in range(p,(n-k)):
#        indice_dep[i] = indice_indep[k-1] + i - p + 1
         
    return indice_indep, indice_dep


# In[5]:


def mod2_multilevel(point):
    
    k = len(point)
    
    for i in range(k):
    
        if point[i]<0:
    
            while point[i] < 0:
                point[i] = point[i] + 2
        else:
        
            point[i] = np.fmod(point[i],2)
    
    return point
        


# In[6]:


def mod4_multilevel(point):
    
    k = len(point)
    
    for i in range(k):
    
        if point[i]<0:
    
            while point[i] < 0:
                point[i] = point[i] + 4
        else:
        
            point[i] = np.fmod(point[i],4)
    
    return point

def mod8_multilevel(point):
    
    k = len(point)
    
    for i in range(k):
    
        if point[i]<0:
    
            while point[i] < 0:
                point[i] = point[i] + 8
        else:
        
            point[i] = np.fmod(point[i],8)
    
    return point


# In[7]:


def modtnn_multilevel(point,shaping_matrix,indice):
    
    if point<0:
    
        while point < 0:
            point = point + shaping_matrix[indice][indice]
    else:
        
        point = np.fmod(point,shaping_matrix[indice][indice])
    
    return point


# In[8]:


# A Python program to print all  
# combinations of given length 
from itertools import combinations 

def combinations_k_l(k,l):


    comb = combinations(range(k), l) 

    comb_vector = [[0 for x in range(l)] for y in range(int(ncr(k, l)))]
    j=0
    
    for p in list(comb): 
        comb_vector[j] = p  
        j+=1
        
    return comb_vector


# In[9]:


#order-l reprocessing (code C1 l = 4)
from itertools import combinations 
#Input: Matrix parity check, vector a and vector z, reprocessing order l
#output: a*
def order_l_reprocessing(G_P,a,z,l):
    #print(a)
    k = len(G_P)
    n = len(G_P[0])
    aux = [0] * n
    
    aux_d = float("inf")
    
    a_est = [0] * n
    
    a_est_bpsk = [0] * n
    
    a_aux = [0] * n
    
    a_est_k = [0] * k
    
    count = 0
    i = 0
    while(i!=l+1): 
        comb = [[0 for x in range(i)] for y in range(int(ncr(k, i)))]
        
        comb = combinations_k_l(k,i) 

        for j in range(int(ncr(k, i))):

        
            a_aux = list(a)
            #print("aa = ", a)
            #print("a_aux = ", a_aux)
            for m in range(i):
                
                #print(comb[j])
                #print("i = ",i)
                a_aux[comb[j][m]] = np.fmod(a_aux[comb[j][m]] + 1,2)
                
                #print(a_aux)
 
            a_est_k = a_aux[:k]              

            a_est = np.fmod(np.matmul(np.transpose(G_P),a_est_k[:]),2)
 
            a_est_bpsk = np.power(-1,a_est[:])
 
            d = np.linalg.norm(a_est_bpsk-z)

            count += 1
            if (d < aux_d):
                   
                aux = a_est[:] 
                aux_d = d

          
            else: 
                pass
        i+=1
 
    return aux,aux_d,count


# In[10]:


#Gram matrix calculation
def gram_mat(generator_matrix, dimension):
    gram_matrix = np.empty(shape=(dimension,dimension))
    gram_matrix = np.matmul(np.transpose(generator_matrix),generator_matrix)
    return gram_matrix


# In[11]:


#Performs a Cholesky decomposition of A, which must 
#be a symmetric and positive definite matrix. The function
#returns the lower variant triangular matrix, L.
def cholesky(A):
    n = len(A)

    # Create zero matrix for L
    L = [[0.0] * n for i in range(n)]

    # Perform the Cholesky decomposition
    for i in range(n):
        for k in range(i+1):
            tmp_sum = sum(L[i][j] * L[k][j] for j in range(k))
            
            if (i == k): # Diagonal elements
                print(A[i][i] - tmp_sum)
                L[i][k] = math.sqrt(A[i][i] - tmp_sum)
                
            else:
                L[i][k] = (1.0 / L[k][k] * (A[i][k] - tmp_sum))
    return L


# In[12]:


#Q matrix calculation of a L matrix in the Cholesky form.
def Q_matrix(L):
    n = len(L)

    # Create zero matrix for Q
    Q = [[0.0] * n for i in range(n)]
    
    # Q coefficient calculation
    for i in range(n):
        Q[i][i] = L[i][i]*L[i][i]
        
    for i in range(n):  
        for j in range(i+1,n):
            Q[i][j]=L[j][i]/L[i][i]; 
    return Q


# In[13]:


#sphere decoding for quantization

from numpy.linalg import inv

def sphere_decoder(r,Q,generator_matrix_inv):
    
    n = len(Q)
    T = [0 for i in range(n)] 
    S = [0 for i in range(n)] 
    L = [0 for i in range(n)]
    u = [0 for i in range(n)]
    eps = [0 for i in range(n)]
    uest = [0 for i in range(n)]
    rho = [0 for i in range(n)] 
    
    #generator_matrix_inv = [[0.0] * n for i in range(n)]
    #generator_matrix_inv = inv(generator_matrix)
    
    rho = generator_matrix_inv.dot(r)
    #print(rho)
    #raio de cobertura do reticulado
    C = 100 #gosset 2^3 * p (p = 1) #BW =  root 2 * p (p = root 3)
    aux = 0
    dest = C
    T[n-1] = C
    for i in range(n):
        S[i] = rho[i]
    ctrl = 1

    while True:
        if ctrl == 1:
            i = n-1
            ctrl = 2
        
        if ctrl == 2:
            
            if T[i]<0:
                T[i] = 0
            L[i] = math.floor(math.sqrt(T[i]/Q[i][i])+S[i])
            u[i] = math.ceil(-math.sqrt(T[i]/Q[i][i])+S[i])-1
            ctrl = 3
            
        if ctrl == 3:
            u[i] = u[i]+1

            if u[i]>L[i]:
                if i == n-1:
                    return dest, uest;
                else:
                    i=i+1
                    ctrl = 3
            else: 
                if i>0:
                    T[i-1] = T[i] - Q[i][i]*((S[i]-u[i])*(S[i]-u[i]))
                    eps[i] = rho[i] - u[i]
                    for j in range (i,n):
                        aux = aux + Q[i-1][j]*eps[j]
                    S[i-1] = rho[i-1] + aux
                    aux = 0
                    i=i-1
                    ctrl = 2
                else:
                    d = T[n-1] - T[0] + Q[0][0]*(S[0]-u[0])*(S[0]-u[0])

                    if d<dest:
                        for i in range(n): 
                            uest[i]=u[i]
                        dest = d
                        T[n-1] = dest
                        ctrl = 1
                    else:
                        ctrl = 3


# In[14]:


def rand_unimod(n):
    l = tril(random.randint(-10, 10, size=(n,n))).astype('float')
    u = triu(random.randint(-10, 10, size=(n,n))).astype('float')
    for i in range(0, n):
        l[i, i] = u[i, i] = 1.0
        if i < n - 1:
            val = sum([l[i, j] * u[j, n-1] for j in range(0, i)])
            u[i, n-1] = (1 - val) / l[i, i]
        else:
            val = sum([l[i, j] * u[j, n-1] for j in range(1, i+1)])
            l[n-1, 0] = (1 - val) / u[0, n-1]
    return dot(l, u), l, u


# In[15]:


def strip_zeros(a):
    """Strip un-necessary leading (rightmost) zeroes
    from a polynomial"""

    return np.trim_zeros(a, trim='b')

def check_type(a, b):
    """Type check and force cast to uint8 ndarray
    Notes
    -----
    Ideally for best performance one should always use uint8 or bool when using this library.
    """

    if isinstance(a, np.ndarray):
        a = np.array(a, dtype="uint8")
    if isinstance(b, np.ndarray):
        b = np.array(b, dtype="uint8")

    if a.dtype is not "uint8":
        a = a.astype("uint8")

    if b.dtype is not "uint8":
        b = b.astype("uint8")

    return a, b

def xor(a, b):
    """Computes the element-wise XOR of two ndarrays"""

    return np.logical_xor(a, b, dtype='uint8').astype("uint8")


def gf2_add(a, b):

    """Add two polynomials in GF(p)[x]
    Parameters
    ----------
    a : ndarray (uint8 or uint8) or list
        Addend polynomial's coefficients.
    b : ndarray (uint8 or uint8) or list
        Addend polynomial's coefficients.
    Returns
    -------
    q : ndarray of uint8
        Resulting polynomial's coefficients.
    Notes
    -----
    Rightmost element in the arrays is the leading coefficient of the polynomial.
    In other words, the ordering for the coefficients of the polynomials is like the one used in MATLAB while
    in Sympy, for example, the leftmost element is the leading coefficient.
    Examples
    ========
    >>> a = np.array([1,0,1], dtype="uint8")
    >>> b = np.array([1,1], dtype="uint8")
    >>> gf2_add(a,b)
    array([0, 1, 1], dtype=uint8)
"""
    a, b = check_type(a, b)

    a, b = strip_zeros(a), strip_zeros(b)

    N = len(a)

    D = len(b)

    if N == D:
        res = xor(a, b)

    elif N > D:

        res = np.concatenate((xor(a[:D], b), a[D:]))

    else:

        res = np.concatenate((xor(a, b[:N]), b[N:]))

    return strip_zeros(res)



def gf2_mul(a, b):
    """Multiply polynomials in GF(2), FFT instead of convolution in time domain is used
       to speed up computation significantly.
    Parameters
    ----------
    a : ndarray (uint8 or bool) or list
        Multiplicand polynomial's coefficients.
    b : ndarray (uint8 or bool) or list
        Multiplier polynomial's coefficients.
    Returns
    -------
    q : ndarray of uint8
        Resulting polynomial's coefficients.
    Examples
    ========
    >>> a = np.array([1,0,1], dtype="uint8")
    >>> b = np.array([1,1,1], dtype="uint8")
    >>> gf2_mul(a,b)
    array([1, 1, 0, 1, 1], dtype=uint8)
"""

    fsize = len(a) + len(b) - 1

    fsize = 2**np.ceil(np.log2(fsize)).astype(int) #use nearest power of two much faster

    fslice = slice(0, fsize)

    ta = np.fft.fft(a, fsize)
    tb = np.fft.fft(b, fsize)

    res = np.fft.ifft(ta*tb)[fslice].copy()

    k = np.mod(np.rint(np.real(res)), 2).astype('uint8')

    return strip_zeros(k)

def gf2_div(dividend, divisor):
    """This function implements polynomial division over GF2.
    Given univariate polynomials ``dividend`` and ``divisor`` with coefficients in GF2,
    returns polynomials ``q`` and ``r``
    (quotient and remainder) such that ``f = q*g + r`` (operations are intended for polynomials in GF2).
    The input arrays are the coefficients (including any coefficients
    equal to zero) of the dividend and "denominator
    divisor polynomials, respectively.
    This function was created by heavy modification of numpy.polydiv.
    Parameters
    ----------
    dividend : ndarray (uint8 or bool)
        Dividend polynomial's coefficients.
    divisor : ndarray (uint8 or bool)
        Divisor polynomial's coefficients.
    Returns
    -------
    q : ndarray of uint8
        Quotient polynomial's coefficients.
    r : ndarray of uint8
        Quotient polynomial's coefficients.
    Notes
    -----
    Rightmost element in the arrays is the leading coefficient of the polynomial.
    In other words, the ordering for the coefficients of the polynomials is like the one used in MATLAB while
    in Sympy, for example, the leftmost element is the leading coefficient.
    Examples
    ========
    >>> x = np.array([1, 0, 1, 1, 1, 0, 1], dtype="uint8")
    >>> y = np.array([1, 1, 1], dtype="uint8")
    >>> gf2_div(x, y)
    (array([1, 1, 1, 1, 1], dtype=uint8), array([], dtype=uint8))
    """

    N = len(dividend) - 1
    D = len(divisor) - 1

    if dividend[N] == 0 or divisor[D] == 0:
        dividend, divisor = strip_zeros(dividend), strip_zeros(divisor)

    if not divisor.any():  # if every element is zero
        raise ZeroDivisionError("polynomial division")
    elif D > N:
        q = np.array([])
        return q, dividend

    else:
        u = dividend.astype("uint8")
        v = divisor.astype("uint8")

        m = len(u) - 1
        n = len(v) - 1
        scale = v[n].astype("uint8")
        q = np.zeros((max(m - n + 1, 1),), u.dtype)
        r = u.astype(u.dtype)

        for k in range(0, m - n + 1):
            d = scale and r[m - k].astype("uint8")
            q[-1 - k] = d
            r[m - k - n:m - k + 1] = np.logical_xor(r[m - k - n:m - k + 1], np.logical_and(d, v))

        r = strip_zeros(r)

    return q, r


# In[ ]:




