# -*- coding: utf-8 -*-
"""

@author: Hristo Goumnerov

******************
*** INCOMPLETE ***
******************

This file computes (one of) the non-dimensional parameter(s)/number(s) from a 
given set of parameters using the Buckingham Pi Theorem.

"""
#%%% Import Modules

# pylint: disable=I0011,C0103,E1101

# Import numpy for mathematical operations
import numpy as np
np.set_printoptions(threshold=np.nan)

# Import sympy to perform matrix inversion using mathematical symbols
import sympy as sy

# Import string to use the latin alphabet
import string

#%%% Input

# Standard physical and material parameters
q = np.array(['M','L','T','θ','E','W','F','P','Q','ρ','E_m','ν_p','U','μ','ν',
              'Pr','σ','c_p','k','α_t','α_m','g'])

# Dimensions [M-mass,L-length,T-time,θ-temperature] of parameters
d = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,2,-2,0],[1,2,-2,0],
              [1,1,-2,0],[1,2,-3,0],[1,0,-3,0],[1,-3,0,0],[1,-1,-2,0],
              [0,0,0,0],[0,1,-1,0],[1,-1,-1,0],[0,2,-1,0],[1,-1,-2,0],
              [1,-1,-2,0],[0,2,-2,-1],[1,1,-3,-1],[0,2,-1,0],[0,0,0,-1],
              [0,1,-2,0]])

# Combined
qd = np.c_[q,d]

##########################
#### Input parameters ####
inp = ['ρ','U','L','μ']
##########################

#%% Create matrices Ax=b

# Assign coefficient matrix
A = np.zeros([len(inp),4], dtype=int)
for i in range(len(inp)):
    A[i,:] = d[np.nonzero(inp[i]==q)[0][0]]

# Remove zeros
A = A[:,~np.all(A==0,axis=0)]

# Transpose matrix
A = A.T

# Assign b vector
b = np.zeros(A.shape[0], dtype=int)
inpp = np.c_[A,b]

# Find reduced row echelon form of symbolic matrix
out = sy.Matrix(inpp).rref()

# Assign x matrix / output
x = np.array(out[0].tolist(), dtype=int)

#%% Print output to console

# Load latin alphabet
alp = list(string.ascii_lowercase)

# Print input and output matrices
print('\nInput:')
print(inp)
print('\nOutput:')
print(x)
print('  ' + ' '.join(str(y) for y in alp[:x.shape[1]]))
print('\n')

# Print resulting parameter equations
for i in range(x.shape[0]):
    
    cur = []
    
    ind = np.nonzero(x[i,:]==1)[0][0]

    for j in range(ind+1,x.shape[1]-1):
        if x[i,j] != 0:
            if x[i,j] == 1:
                cur.append('-' + alp[j])
            elif x[i,j] == -1:
                cur.append(alp[j])
            else:
                cur.append(str(-x[i,j])+alp[j])
        
    cur2 = ' + '.join(str(y) for y in cur)
    
    dv = alp[ind]
    
    stout = dv + ' = ' + str(x[i,-1]) + ' - ' + cur2
    if x[i,-1] == 0:
        stout = dv + ' = ' + cur2
    
    print(stout + '\n')
#%% Print out final non-dimensional number

output = []
out1 = []
for j in range(len(inp)):
    out1.append(str(inp[j]+'^'+alp[j]))

print('*'.join(out1))
