#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Question 1


# In[ ]:


import numpy as np
import math
from random import randint
def main():
    a=np.zeros((100,100))
    b=numpy.zeros(100)
    n=100
    for i in range(100):
        for j in range(100):
            a[i][j]=randint(0,10)
        b[i]=randint(0,10)
    gauss(a,b)

def forward_elimination(A, b, n):
    """
    Calculates the forward part of Gaussian elimination.
    """
    for row in range(0, n-1):
        for i in range(row+1, n):
            factor = A[i,row] / A[row,row]
            for j in range(row, n):
                A[i,j] = A[i,j] - factor * A[row,j]

            b[i] = b[i] - factor * b[row]

        print('A = \n%s and b = %s' % (A,b))
    return A, b

def back_substitution(a, b, n):
    """"
    Does back substitution, returns the Gauss result.
    """
    x = np.zeros((n,1))
    x[n-1] = b[n-1] / a[n-1, n-1]
    for row in range(n-2, -1, -1):
        sums = b[row]
        for j in range(row+1, n):
            sums = sums - a[row,j] * x[j]
        x[row] = sums / a[row,row]
    return x

def gauss(A, b):
    """
    This function performs Gauss elimination without pivoting.
    """
    n = A.shape[0]

    # Check for zero diagonal elements
    if any(np.diag(A)==0):
        raise ZeroDivisionError(('Division by zero will occur; '
                                  'pivoting currently not supported'))

    A, b = forward_elimination(A, b, n)
    return back_substitution(A, b, n)
main()


# In[ ]:


#Question 2


# In[5]:


import numpy as np
from random import randint
def gaussian_elimination_with_pivot(m):

  n = 1000
  for i in range(n):
    pivot(m, n, i)
    for j in range(i+1, n):
      m[j] = [m[j][k] - m[i][k]*m[j][i]/m[i][i] for k in range(n+1)]

  if m[n-1][n-1] == 0: raise ValueError('No unique solution')

  # backward substitution
  x = [0] * n
  for i in range(n-1, -1, -1):
    s = sum(m[i][j] * x[j] for j in range(i, n))
    x[i] = (m[i][n] - s) / m[i][i]
  return x

def pivot(m, n, i):
  max = -1e100
  for r in range(i, n):
    if max < abs(m[r][i]):
      max_row = r
      max = abs(m[r][i])
  m[i], m[max_row] = m[max_row], m[i]

def main():
  m = np.zeros((100,100))
  for i in range(100):
     for j in range(100):
        m[i][j]=randint(0,10)
  print(gaussian_elimination_with_pivot(m))
main()


# In[ ]:


#Question 3


# In[ ]:


import math
import numpy as np
from random import randint
def LU (table): 
   
    rows,columns=np.shape(table)
    L=np.zeros((rows,columns))
    U=np.zeros((rows,columns))
    if rows!=columns:
        return
    for i in range (columns):
        for j in range(i-1):
            sum=0
            for k in range (j-1):
                sum+=L[i][k]*U[k][j]
            L[i][j]=(table[i][j]-sum)/U[j][j]
        L[i][i]=1
        for j in range(i-1,columns):
            sum1=0
            for k in range(i-1):
                sum1+=L[i][k]*U[k][j]
            U[i][j]=table[i][j]-sum1
    return L,U



matrix =np.zeros((100,100))
for i in range(100):
        for j in range(100):
            matrix[i][j]=randint(0,10)
        
    
L,U = LU(matrix)
print(L)
print(U)

