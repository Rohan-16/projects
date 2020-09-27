#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import matplotlib.pyplot as plt

#This function fits a linear function to the data. The data
#is stored in matrix variable f where the 1st column is 
#attribute values and the 2nd column is target values. The
#function returns the co-efficients

def linear(f):
	s = np.size(f[:,0]) #number of sample observations
	x = f[:,0]          #attribute values: a row vector
	t = f[:,1]          #target values: a row vector
	u = np.ones((1,s))  #a row vector of all 1s
	u = np.vstack((u,x))#the transpose of the data matrix shown in the class
	v = np.linalg.inv(u@u.T) #v = (uu')^-1
	w = v@u@t           #w = vut
	return(w)

#This function plots the linear function with coefficient vector w
#against the attribue values. It also plots the data as the dots.
#The data is stored in matrix variable f where the 1st column is
#attribute values and the 2nd column is target values. 

def plotit(w, f):
        s = np.size(f[:,0])#These five lines are the same as in linear(f)
        x = f[:,0]
        t = f[:,1]
        u = np.ones((1,s))
        u = np.vstack((u,x))
        t1 = w@u
        plt.plot(x,t,'o', x,t1)
        plt.xlabel('years')
        plt.ylabel('winning times')
        plt.xlim(1920,2020)
        plt.show()
def main():
    input = np.loadtxt("Olympic_women.txt", dtype='f', delimiter=',')
    print(input) 
    w=linear(input)
    print(linear(input))
    plotit(w,input)
    twentytwelve=(((w[1])*2012)+w[0])
    twentysixteen=(((w[1])*2016)+w[0])
    print("Prediction for 2012 is ",twentytwelve,"s")
    print("Prediction for 2016 is ",twentysixteen,"s")
main()


# In[ ]:




