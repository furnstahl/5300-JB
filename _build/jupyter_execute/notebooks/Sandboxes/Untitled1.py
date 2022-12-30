#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from numpy.random import uniform, rand

from scipy.integrate import solve_ivp
from scipy.spatial import ConvexHull
from scipy.linalg import orth
from scipy.linalg import eigh

import matplotlib.pyplot as plt


# In[2]:


U = np.random.uniform(low=0, high=1.0, size=(10, 10))
H_test = np.tril(U) + np.tril(U, -1).T


# In[6]:


eigvals, eigvecs = eigh(H_test)


# In[9]:


print(eigvals)


# In[10]:


X = np.random.uniform(low=0, high=1.0, size=(10, 10))


# In[11]:


eigvals2, eigvecs2 = eigh(X.T @ H_test @ X)


# In[12]:


print(eigvals2)


# In[ ]:




