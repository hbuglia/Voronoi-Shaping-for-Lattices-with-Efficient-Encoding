#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Voronoi Shaping for Lattices with Efficient Encoding
#Section V-A - Extended BCH Lattice Codes
#Shaping with E8 Lattice

#Authors: Henrique Buglia and Renato da Rocha Lopes


# In[2]:


import struct
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot, pylab  
import scipy.spatial
import random
from numpy.linalg import inv


# In[3]:


import nbimporter
from Functions import ncr
from Functions import gf2elim
from Functions import Linear_Independence
from Functions import mod2_multilevel
from Functions import mod4_multilevel
from Functions import mod8_multilevel
from Functions import combinations_k_l
from Functions import order_l_reprocessing
from Functions import gram_mat
from Functions import cholesky
from Functions import Q_matrix
from Functions import sphere_decoder
from Functions import modtnn_multilevel
from Functions import rand_unimod
from Functions import strip_zeros
from Functions import check_type
from Functions import xor
from Functions import gf2_add
from Functions import gf2_mul
from Functions import gf2_div


# In[4]:


#C1 contido em C2

n = 127
#generate BCH Codes C1 (128,78,7) 
#g1 = [1,0,1,1,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,1,1,0,0,0,1,0,1,1,0,0,0,0,0,1,0,0,1,1,0,1]
g1 = [1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,1,1,0,1,1,1,0,0,1,1,1,0,0,0,1,1]
#generate BCH Codes C2 (128,120,1)    
g2 = [1,0,0,0,1,0,0,1]

#C1 Generator Matrix G_gen_1[k_1 x n]
k_1 = 78

G_gen_1 = [[0 for x in range(k_1)] for y in range(n)] 


for i in range(k_1):
        
    for j in range(n-k_1+1):
        G_gen_1[j+i][i] = g1[j] 
        
#C1 Generator Matrix in systematic form G_1[k_1 x n]        
        
G_1_t = gf2elim(np.transpose(G_gen_1))    
G_1 = np.transpose(G_1_t) 


#C2 Generator Matrix G_gen_2[k_2 x n]
k_2 = 120

G_gen_2 = [[0 for x in range(k_2)] for y in range(n)]
for i in range(k_2):
    for j in range(n-k_2+1):
        G_gen_2[j+i][i] = g2[j] 

#C2 Generator Matrix in systematic form G_2[k_2 x n]        
        
G_2_t = gf2elim(np.transpose(G_gen_2))    
G_2 = np.transpose(G_2_t) 


# In[5]:


#shaping operation

rectangular_shaping = [[8,0,0,0,0,0,0,0],
                      [0,8,0,0,0,0,0,0],
                      [0,0,8,0,0,0,0,0],
                      [0,0,0,8,0,0,0,0],
                      [0,0,0,0,8,0,0,0],
                      [0,0,0,0,0,8,0,0],
                      [0,0,0,0,0,0,8,0],
                      [0,0,0,0,0,0,0,8]]

rectangular_shaping_inv = inv(rectangular_shaping)

#gosset
shaping_dimension = 8
l_shaping = int((128/shaping_dimension))

E8_Generator_Matrix = [[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],
                      [0,1,0,0,0,0,0,1],
                      [0,0,1,0,0,0,0,1],
                      [0,0,0,1,0,0,0,1],
                      [0,0,0,0,1,0,0,1],
                      [0,0,0,0,0,1,0,1],
                      [0,0,0,0,0,0,1,1],
                      [0,0,0,0,0,0,0,2]]


Integer_shap_Generator_Matrix = (2*np.array(E8_Generator_Matrix)).astype(int)

Shaping_Lattice_Generator_Matrix = (4*np.array(Integer_shap_Generator_Matrix)).astype(int)

Shaping_Lattice_Generator_Matrix_inv = inv(Shaping_Lattice_Generator_Matrix)

#FOR SPHERE DECODER

#G = gram_mat(np.transpose(rectangular_shaping),8)

G = gram_mat(np.transpose(Shaping_Lattice_Generator_Matrix),8)

G_rec = gram_mat(np.transpose(rectangular_shaping),8)
#performing cholesky decomposition for Sphere decoder

R = np.linalg.cholesky(G)

R_rec = np.linalg.cholesky(G_rec)

#Calculating Q matrix for Sphere Decoder

Q = Q_matrix(R)

Q_rec = Q_matrix(R_rec)


# In[6]:


def transmition_sys(sigma,mmse):
    

    q=2

    #ENCODING
    
    #generating a random message u1

    u_1 = np.random.randint(q, size=k_1)

    
     #systematic encoding via polynomials C1
    
    x_k_n_1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    
    shift_m_poly = gf2_mul(np.array(u_1[::-1]), np.array(x_k_n_1))
 
    q_poly, r_poly = gf2_div(np.array(shift_m_poly), np.array(g1[::-1]))
    
    c_1_syst_reverse = gf2_add(np.array(shift_m_poly),np.array(r_poly))

    c_1 = n * [0] 

    
    for i in range(len(c_1_syst_reverse)):
        c_1[i + n-len(c_1_syst_reverse)] = c_1_syst_reverse[len(c_1_syst_reverse)-i-1]


    
    #c_1_mat = np.fmod(np.dot(G_1, u_1),q)


    
    Parity_c1 = np.fmod(scipy.spatial.distance.hamming(c_1,[0]*n)*n,2).astype(int)

    c_1_extended = list(c_1) + [Parity_c1]


        #generating a random message u2

    u_2 = np.random.randint(q, size=k_2)
    
    #systematic encoding via polynomials C2
    
    x_k_n_2 = [0,0,0,0,0,0,0,1]
    
    shift_m_poly_2 = gf2_mul(np.array(u_2[::-1]), np.array(x_k_n_2))
 
    q_poly_2, r_poly_2 = gf2_div(np.array(shift_m_poly_2), np.array(g2[::-1]))
    
    c_2_syst_reverse = gf2_add(np.array(shift_m_poly_2),np.array(r_poly_2))
    
    c_2 = n * [0] 
    
    for i in range(len(c_2_syst_reverse)):
        c_2[i + n - len(c_2_syst_reverse)] = c_2_syst_reverse[len(c_2_syst_reverse)-i-1]
        
    #c_2_mat = np.fmod(np.dot(G_2, u_2),q)

    
    Parity_c2 = np.fmod(scipy.spatial.distance.hamming(c_2,[0]*n)*n,2).astype(int)

    c_2_extended = list(c_2) + [Parity_c2]
    #generate a random element of S set

    random_s = [0] * 128
    
    for j in range(16):
        for i in range(8):
            random_s[i+8*j] = random.randint(0,Integer_shap_Generator_Matrix[i][i]-1)

        #generate a random lattice point

    x = np.array(c_1_extended) + 2*np.array(c_2_extended) + (q**2)*np.array(random_s)
    
    #applyng shaping operation
    integer_cordinates = [0] * 8
    shaping_lattice_point = [0] * 128
    
    for i in range(16):
        
        d, integer_cordinates = sphere_decoder(list(x[i*8:i*8 + 8]),Q,np.transpose(Shaping_Lattice_Generator_Matrix_inv))
        
        shaping_lattice_point[i*8:i*8 + 8] = np.transpose(Shaping_Lattice_Generator_Matrix).dot(integer_cordinates)
        
    
    x_coset = x - shaping_lattice_point
    
    #TRANSMISSION THROUGH THE THE CHANNEL

    w = np.random.normal(0,sigma,128)

    #received point
    y_channel = mmse*(x_coset + w)

    y = y_channel[0:127]
    ##Decoding y1_bpsk = c_1_bpsk + w mod 2
    
    y1 = abs(mod2_multilevel(y+1)-1) 


    #translating the lattice in order to obtain a bpsk transmition

    y1_bpsk = -2*y1  + 1
    
    
    #DECODING
    

    ##OSM to find c_1_bpsk_est

    y1_abs = np.absolute(y1_bpsk)

    #reeodering in descending order
    y1_asc = np.sort(y1_abs)
    y1_desc = sorted(y1_asc,reverse=True)
    #getting permutation indices 1 
    indices1 = np.argsort(y1_abs)

    #reordering indices 1 in descending order

    indices1_reordered = [0] * n

    for i in range (n):
        indices1_reordered[i] = indices1[n-i-1]

    #creating the corresponding permutation matrix P1 
    P1 = [[0 for x in range(n)] for y in range(n)]

    for i in range(n):
        P1[i][indices1_reordered[i]] = 1

    #Obtaining the permutation matrix G_1_P1
    G_1_P1 = np.matmul(P1, G_1)

    #getting permutation indices 2

    indices2_indep, indices2_dep = Linear_Independence(G_1_P1,k_1,n)

    #creating the corresponding permutation matrix P2 
    P2 = [[0 for x in range(n)] for y in range(n)]

    for i in range(k_1):
        P2[i][indices2_indep[i]] = 1


    for i in range(k_1,n):
        P2[i][indices2_dep[i-k_1]] = 1

    #Obtaining the permutation matrix G_1_P2
    G_1_P2 = np.matmul(P2, G_1_P1)

    #Obtaining the total permutation matrix P

    P = np.matmul(P2, P1)

    #Obtaining the total permutation sequence z

    z = np.matmul(P,y1_bpsk)

    #Gaussian elimination in GF(2)
    G1 = gf2elim(np.transpose(G_1_P2))


    #Performing hard decision for the first k bits
    b_k1 = [0] * k_1 

    for i in range (k_1):
        if z[i] < 0:
            b_k1[i] = 1
        else:
            b_k1[i] = 0

    #obtaining the hard-decision-decoded codeword 

    b = [0] * n

    b = np.fmod(np.matmul(np.transpose(G1),b_k1),q)

    #order-l reprocessing

    a = [0] * n

    a, d, count = order_l_reprocessing(G1,b[:],z[:],3)

    #reordering

    c_1_est = np.fmod(np.matmul(np.transpose(P),a),q)

    #obtaining u_1_est

    u_1_est = c_1_est[:k_1]

    #Estimated codeword C1

    c1_est = np.fmod(np.matmul(G_1,u_1_est),2)

    Parity_c1_est = np.fmod(scipy.spatial.distance.hamming(c_1_est,[0]*n)*n,2).astype(int)
    
    c_1_extended_est = list(c_1_est) + [Parity_c1_est]
    ##Decoding y2 = c_2_bpsk + w mod 2

    y2 = abs(mod2_multilevel(((y - c1_est)/q)+1)-1)

    #translating the point in ordem to obtain a bpsk

    y2_bpsk = -2*y2 + 1

    ##OSM to find c_2_bpsk_est
    
    y2_abs = np.absolute(y2_bpsk)

    #reeodering in descending order

    y2_asc = np.sort(y2_abs)
    y2_desc = sorted(y2_asc,reverse=True)

    #getting permutation indices 1 

    indices1_2 = np.argsort(y2_abs)

    #reordering indices 1 in descending order

    indices1_2_reordered = [0] * n

    for i in range (n):
        indices1_2_reordered[i] = indices1_2[n-i-1]

    #creating the corresponding permutation matrix P1 
    P1_2 = [[0 for x in range(n)] for y in range(n)]

    for i in range(n):
        P1_2[i][indices1_2_reordered[i]] = 1

    #Obtaining the permutation matrix G_1_P1
    G_2_P1 = np.matmul(P1_2, G_2)

    #getting permutation indices 2

    indices2_2_indep, indices2_2_dep = Linear_Independence(G_2_P1,k_2,n)

    #creating the corresponding permutation matrix P2 
    P2_2 = [[0 for x in range(n)] for y in range(n)]

    for i in range(k_2):
        P2_2[i][indices2_2_indep[i]] = 1


    for i in range(k_2,n):
        P2_2[i][indices2_2_dep[i-k_2]] = 1

    #Obtaining the permutation matrix G_1_P2
    G_2_P2 = np.matmul(P2_2, G_2_P1)

    #Obtaining the total permutation matrix P

    P_2 = np.matmul(P2_2, P1_2)

    #Obtaining the total permutation sequence z

    z_2 = np.matmul(P_2,y2_bpsk)

    #Gaussian elimination in GF(2)
    G1_2 = gf2elim(np.transpose(G_2_P2))



    #Performing hard decision for the first k bits
    b_k2 = [0] * k_2 

    for i in range (k_2):
        if z_2[i] < 0:
            b_k2[i] = 1
        else:
            b_k2[i] = 0

    #obtaining the hard-decision-decoded codeword 

    b2 = [0] * n

    b2 = np.fmod(np.matmul(np.transpose(G1_2),b_k2),q)


    #order-l reprocessing

    a2 = [0] * n

    a2, d2, count2 = order_l_reprocessing(G1_2,b2[:],z_2[:],1)

    #reordering

    c_2_est = np.fmod(np.matmul(np.transpose(P_2),a2),q)

    #obtaining u_1_est

    u_2_est = c_2_est[:k_2]

    #Estimated codeword C2

    c2_est = np.fmod(np.matmul(G_2,u_2_est),2)
    
    
    Parity_c2_est = np.fmod(scipy.spatial.distance.hamming(c_2_est,[0]*n)*n,2).astype(int)
    
    c_2_extended_est = list(c_2_est) + [Parity_c2_est]
    #Estimating the random integer which put the point inside shaping lattice 

    random_integer_est = np.rint((y_channel - np.array(c_1_extended_est) - q*np.array(c_2_extended_est))/4).astype(int)

    #Estimating the transmitted lattice point inside shaping lattice
    x_est = np.array(c_1_extended_est) + q*np.array(c_2_extended_est) + (q**2)*random_integer_est

    
    #Estimating the random s of set S 
    
    r = (x_est - np.array(c_1_extended_est) - q*np.array(c_2_extended_est))/(q**2)
    
    random_s_est = [0] * 128
    z_integer = [0] * shaping_dimension
    aux = 0

    for j in range(l_shaping):
        for i in range(shaping_dimension):
            aux = r[j*shaping_dimension + i]
            for k in range(i):
                aux = aux + (Integer_shap_Generator_Matrix[i][k])*z_integer[k]
            random_s_est[j*shaping_dimension + i] = modtnn_multilevel(aux,Integer_shap_Generator_Matrix,i)
            z_integer[i] = ((random_s_est[j*shaping_dimension + i] - aux)/Integer_shap_Generator_Matrix[i][i])
  
    #Estimating the transmitted lattice point outside shaping lattice

    x_est_out = np.array(c_1_extended_est) + q*np.array(c_2_extended_est) + (q**2)*np.array(random_s_est)
    #print(x_est_out)
    #calculating number of errors
    
    #n_errors = scipy.spatial.distance.hamming(x_est_out,x)*(n+1)
    n_errors = scipy.spatial.distance.hamming(x_est,x_coset)*(n+1)
    if(n_errors):
        
        error = 1
        
    else:
        
        error = 0
    
    return n_errors, error


# In[7]:


num_erros = 5
passo = 0.5
Eb_N0_db = 14.5
limite_Eb_N0_db = 19

#gosset average power per 2 dim (P = 9.1716 if we use continous aprox)

P = 9.1716

#Cubic Shaping average power

#P = 10.5


# In[8]:


total_SER_errors = 0
total_WER_errors = 0
i = 0

while(Eb_N0_db < limite_Eb_N0_db):
    
    Eb_N0_db = Eb_N0_db + passo
    
    sd = (P/(2*(10**(Eb_N0_db/10))))**(1/2)
    
    mmse_coef =P/(P + (sd**2))
    
    if(Eb_N0_db < 17):
        num_erros = 20
    else:
        num_erros = 5
    
    while(total_WER_errors < num_erros):
        
        errors_SER, errors_WER = transmition_sys(sd,mmse_coef)
        
        total_SER_errors = total_SER_errors + errors_SER
        
        total_WER_errors = total_WER_errors + errors_WER
        
        i += 1

        
    SER = total_SER_errors/(128*i) 
    WER = total_WER_errors/i
    print("EbN0db = ", Eb_N0_db, "SER = ", SER, "WER = ", WER)
    total_SER_errors = 0
    total_WER_errors = 0
    i = 0


# In[54]:


total_SER_errors/(128*i)


# In[9]:


total_WER_errors


# In[ ]:


total_WER_errors/i


# In[ ]:


total_SER_errors/(128*i)


# In[ ]:




