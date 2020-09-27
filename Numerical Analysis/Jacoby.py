#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np 
from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot

def jacobi(A,b,N=100,x=None,tol=1e-15):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed 
    a=0
    if x is None:
        x = zeros(len(A[0]))

    # Create a vector of the diagonal elements of A                                                                                                                                                
    # and subtract them from A                                                                                                                                                                     
    D = diag(A)
    R = A - diagflat(D)

        # Iterate for N times                                                                                                                                                                          
    for i in range(N):
        x2 = (b - dot(R,x)) / D
        delta = np.linalg.norm(x - x2)
        if delta < tol:
            return x2
        x = x2

    warnings.warn(f"did not converge within {N} iterations")
    

    return x

A = np.array([[3.0, 1.0, 0., 0., 0., 0., 0., 0., 0., 0.],[1.0, 3.0, 1.0, 0., 0., 0., 0., 0., 0., 0.], [0., 1.0, 3.0, 1.0, 0., 0., 0., 0., 0., 0.], [0., 0, 1.0, 3.0, 1.0, 0., 0., 0., 0., 0.], [0., 0., 0., 1.0, 3.0, 1.0, 0., 0., 0., 0.], [0., 0., 0., 0., 1.0, 3.0, 1.0, 0., 0., 0.], [0., 0., 0., 0., 0., 1.0, 3.0, 1.0, 0., 0.], [0., 0., 0., 0., 0., 0., 1.0, 3.0, 1.0, 0.], [0., 0., 0., 0., 0., 0., 0., 1.0, 3.0, 1.0], [0., 0., 0., 0., 0., 0., 0., 0., 1.0, 3.0]])
b = np.array([1.0,1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
guess = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

sol = jacobi(A,b,N=100,x=guess)

print ("A:")
pprint(A)

print ("b:")
pprint(b)

print ("x:")
pprint(sol)


# In[11]:


from matrix import height

def gauss_seidel(m, x0=None, eps=1e-5, max_iteration=100):
  
  n  = height(m)
  x0 = [0] * n if x0 == None else x0
  x1 = x0[:]

  for __ in range(max_iteration):
    for i in range(n):
      s = sum(-m[i][j] * x1[j] for j in range(n) if i != j) 
      x1[i] = (m[i][n] + s) / m[i][i]
    if all(abs(x1[i]-x0[i]) < eps for i in range(n)):
      return x1 
    x0 = x1[:]    
  raise ValueError('Solution does not converge')

if __name__ == '__main__':
  m =([[3.0, 1.0, 0., 0., 0., 0., 0., 0., 0., 0.],[1.0, 3.0, 1.0, 0., 0., 0., 0., 0., 0., 0.], [0., 1.0, 3.0, 1.0, 0., 0., 0., 0., 0., 0.], [0., 0, 1.0, 3.0, 1.0, 0., 0., 0., 0., 0.], [0., 0., 0., 1.0, 3.0, 1.0, 0., 0., 0., 0.], [0., 0., 0., 0., 1.0, 3.0, 1.0, 0., 0., 0.], [0., 0., 0., 0., 0., 1.0, 3.0, 1.0, 0., 0.], [0., 0., 0., 0., 0., 0., 1.0, 3.0, 1.0, 0.], [0., 0., 0., 0., 0., 0., 0., 1.0, 3.0, 1.0], [0., 0., 0., 0., 0., 0., 0., 0., 1.0, 3.0]])
  print(gauss_seidel(m))






# In[14]:


import numpy as np 
import math 

A = np.array([[3.0, 1.0, 0., 0., 0., 0., 0., 0., 0., 0.],[1.0, 3.0, 1.0, 0., 0., 0., 0., 0., 0., 0.], [0., 1.0, 3.0, 1.0, 0., 0., 0., 0., 0., 0.], [0., 0, 1.0, 3.0, 1.0, 0., 0., 0., 0., 0.], [0., 0., 0., 1.0, 3.0, 1.0, 0., 0., 0., 0.], [0., 0., 0., 0., 1.0, 3.0, 1.0, 0., 0., 0.], [0., 0., 0., 0., 0., 1.0, 3.0, 1.0, 0., 0.], [0., 0., 0., 0., 0., 0., 1.0, 3.0, 1.0, 0.], [0., 0., 0., 0., 0., 0., 0., 1.0, 3.0, 1.0], [0., 0., 0., 0., 0., 0., 0., 0., 1.0, 3.0]])
b = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
x0 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
tol =  10 ** (-15)
max_iter = 20
w = 1.5

def SOR(A, b, x0, tol, max_iter, w): 
    if (w<=1 or w>2): 
        print('w should be inside [1, 2)'); 
        step = -1; 
        x = float('nan') 
        return 
    n = b.shape 
    x = x0 

    for step in range (1, max_iter): 
        for i in range(n[0]): 
            new_values_sum = np.dot(A[i, :i], x[:i])
            old_values_sum = np.dot(A[i, i+1 :], x0[ i+1: ]) 
            x[i] = (b[i] - (old_values_sum + new_values_sum)) / A[i, i] 
            x[i] = np.dot(x[i], w) + np.dot(x0[i], (1 - w))  
        #if (np.linalg.norm(x - x0) < tol): 
        if (np.linalg.norm(np.dot(A, x)-b ) < tol):
            print(step) 
            break 
        x0 = x

    print("X = {}".format(x)) 
    print("The number of iterations is: {}".format(step))
    return x
x = SOR(A, b, x0, tol, max_iter, w)
print(np.dot(A, x))


# In[ ]:




