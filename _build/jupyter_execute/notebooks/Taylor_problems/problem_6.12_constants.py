#!/usr/bin/env python
# coding: utf-8

# # Finding constants using fsolve

# Problem: find $k$ and $c$ given
# 
# $\begin{align}
#   1 + \sinh^{-1}1 &= k \sinh^{-1}(1/k) + c \;,  \\
#   1 + \sinh^{-1}5 &= k \sinh^{-1}(5/k) + c \;.  
# \end{align}$
# 
# Plan: use `fsolve` from `scipy.optimize`.

# In[1]:


import numpy as np
from scipy.optimize import fsolve


# In[2]:


def func(x):
    """Function of x = (k, c) defined so that when each component is zero we 
        have our solution.  
       No extra arguments need to be passed, so func is simple."""
    k, c = x
    return (
            1. + np.arcsinh(1.) - (k * np.arcsinh(1./k) + c),
            1. + np.arcsinh(5.) - (k * np.arcsinh(5./k) + c)
           )


# In[3]:


x0 = (0.1, 0.1)   # guesses for k and c
k, c = fsolve(func, x0)
print(f'k = {k:0.2f}, c = {c:0.2f}')


# In[ ]:




