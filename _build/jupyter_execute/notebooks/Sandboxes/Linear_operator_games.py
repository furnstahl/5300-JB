#!/usr/bin/env python
# coding: utf-8

# # Demo notebook for linear operators as matrices
# 
# In this notebook we explore implementing linear operators in Python as matrices (using numpy).

# In[1]:


import numpy as np


# First make a vector with specified spacing:

# In[2]:


t = np.arange[0,10,1]


# Oops, I thought I was using Mathematica. Fix the brackets and check that we got what we wanted:

# In[3]:


t = np.arange(0,10,1)
print(t)


# Useful matrix functions include np.identity and np.ones:

# In[4]:


print( np.identity(5) )


# In[5]:


print( np.ones(5) )


# In[6]:


print( np.ones((5,5)) )


# Ok, now try an operator that multiplies by $\omega_0^2$:

# In[7]:


omega0 = 2
D_op = omega0**2 * np.identity(10)
print(D_op)


# Try it out! It is tempting to use `*` but for linear algebra we use `@`. Compare:

# In[8]:


print(D_op * t)


# In[9]:


print(D_op @ t)


# Ok, let's make it more general (note the difference between these two):

# In[10]:


t_min = 0
t_max = 10
delta_t = 1
t = np.arange(t_min, t_max, delta_t)
print(t)


# In[11]:


t_min = 0
t_max = 10
delta_t = 1
t = np.arange(t_min, t_max, delta_t)
print(t)
num_t = len(t)
print(num_t)


# In[12]:


omega0 = 2
D_op = omega0**2 * np.identity(num_t)
print(D_op)


# Now try the simplest derivative operator, building it from shifted diagonal matrices.

# In[13]:


print( np.diag(np.ones(5), 0) )


# In[14]:


print( np.diag(np.ones(5), 1) )


# In[15]:


print( np.diag(np.ones(5), -1) )


# In[16]:


Diff_op = (1 * np.diag(np.ones(num_t-1), 1) + (-1) * np.diag(np.ones(num_t), 0)) / delta_t
print(Diff_op)


# Try it!

# In[17]:


print(Diff_op @ t)


# In[18]:


print(Diff_op @ t**2, '\n', 2*t)


# Build a better derivative operator by making it *symmetric*:

# In[19]:



Diff_sym_op = (1 * np.diag(np.ones(num_t-1), 1) + (-1) * np.diag(np.ones(num_t-1), -1)) / (2*delta_t)
print(Diff_sym_op)


# In[20]:


print(Diff_sym_op @ t**2, '\n', 2*t)


# In[21]:


print(Diff_sym_op @ t**3, '\n', 3*t**2)


# Try with smaller spacing `delta_t` to get more accuracy.

# In[ ]:




