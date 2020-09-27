#!/usr/bin/env python
# coding: utf-8

# In[7]:


#!/usr/bin/env python
# coding: utf-8

# In[41]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt



#Function to read the file with the data.
def main():

    attributes= np.loadtxt("attributes.txt")
    labels= np.loadtxt("targets.txt")
    smallest=10000
    smallestk=1000000
    
    for i in range(1,30):
        
        x=(CrossVal10(attributes,labels,i))
        if (x<smallest):
            smallest=x
            smallestk=i
    print("The best k is ", smallestk,"with CV error",smallest)
    arr=[]
    array1=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    for i in range(1,31):
        arr.append((CrossVal10(attributes,labels,i))/10)
    plt.plot(array1,arr)
    plt.title("%age misclassifications versus all K from 1 through 30")
    plt.xlabel("K")
    plt.ylabel("Percentage misclassifications")
    
    
        

# Perform 10-fold cross-validation for KNN. 
# m is the matrix in which rows are for objects and columns are attribute values
# c is a single column matrix of labels.
# k is the neighbor size for KNN.
def CrossVal10(m, c, k):
    size = np.size(m, axis=0) 
    fsize = size//k   
    er = 0
    for i in range(k):
        trainx = np.delete(m, range(i*fsize, (i+1)*fsize), axis=0)
        
        trainl = np.delete(c, range(i*fsize, (i+1)*fsize))
        
        testx = m[i*fsize:(i+1)*fsize,:]
        
        testl = c[i*fsize:(i+1)*fsize]
        
        knn= KNN(trainx,trainl,testx,k)
        error=0
        for i in range(np.size(knn)):
            if(knn[i] != testl[i]):
                error = error + 0.1
            

        er= er + error

    return(er)

    #add the code below to do the following: 1. call your KNN function.
    #It should be something like KNN(trainx, trainl, testx, k); 2. compute
    #the classification error for the current fold; 3. return the
    #average classification rate over all folds. 

        
    

#Use KNN with neighbor size of k to classify each row in testx. 
# The training set is trainx and trainl, which contain attributes and labels.
#Returns a vector of labels for the rows in testx.

def KNN(trainx, trainl, testx, k):
    
    labels=[]
    for i in range(np.size(testx,axis=0)): 
        
        
        m=[testx[i,:]]
        elementI= np.array(m)

    

        repeatAr= np.repeat(elementI,np.size(trainx,axis=0),axis=0)

        distAr= np.sum((trainx-repeatAr)**2,axis=1)

        ind= np.argsort(distAr)

        kNeighbors= ind[0:k]

        classACount= 0
        classBCount= 0
        finalLabel= None
        for i in range(np.size(kNeighbors,axis=0)):
            targetValue= trainl[kNeighbors[i]]
            if(targetValue==0):
                classACount= classACount+1
            else:
                classBCount= classBCount+1
        if(classACount>classBCount): 
            
            finalLabel= 0.
            labels.append(finalLabel)
        else:
            finalLabel=1.
            labels.append(finalLabel)
    
    return np.array(labels)
   

        

            
        
        
        
   


        

main()


# In[ ]:


# In[ ]:


# In[ ]:




