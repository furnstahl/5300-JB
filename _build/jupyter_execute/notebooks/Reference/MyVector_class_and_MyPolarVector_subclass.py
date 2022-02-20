#!/usr/bin/env python
# coding: utf-8

# # Example of class and subclass
# 
# Adapted from https://www.astro.umass.edu/~schloerb/ph281/Lectures/OOP/OOP.pdf.

# In[ ]:


import numpy as np


# In[ ]:


class MyVector:
    '''Demo Class to manage vector and operations'''
    
    def __init__(self, x, y, z):
        '''constructor'''
        self.x = x
        self.y = y
        self.z = z
    
    def __str__(self):
        '''makes printable representation of vector'''
        return f'MyVector({self.x:f}, {self.y:f}, {self.z:f})'
    
    def __add__(self, other):
        '''adds vector'''
        return MyVector(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        '''subtracts vector'''
        return MyVector(self.x-other.x,self.y-other.y,self.z-other.z)
    
    def __mul__(self, scalar):
        '''multiplies vector by scalar'''
        return MyVector(scalar * self.x,scalar * self.y,scalar * self.z)
    
    def __div__(self, scalar):
        '''divides vector by scalar'''
        return MyVector(self.x/scalar,self.y/scalar,self.z/scalar)
    
    def norm(self):
        '''computes magnitude of vector'''
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def unit(self):
        '''creates a unit vector'''
        return self / self.norm()
    
    def dot(self, other):
        '''computes dot product'''
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self,other):
        '''computes cross product'''
        new_x = self.y * other.z - self.z * other.y
        new_y = self.z * other.x - self.x * other.z
        new_z = self.x * other.y - self.y * other.x
        return MyVector(new_x, new_y, new_z)


# In[ ]:


a = MyVector(3., 1., 0.)
print(a)


# In[ ]:


b = MyVector(2., 2., 2.)
print(b)


# In[ ]:


print(f'Norm of a is {a.norm()}')
print(f'Dot product of a and b is {a.dot(b)}')
print(f'Cross product of a and b is {a.cross(b)}')
print(f'Sum of a and b is {a+b}')


# Now introduce a new class `MyPolarVector` that inherits the methods from `MyVector`.

# In[ ]:


class MyPolarVector(MyVector):  # subclass of MyVector
    '''vector in polar coordinates'''
    def __init__(self, r, theta, phi):
        '''creates a MyVector instance'''
        MyVector.__init__(self,
                          r * np.cos(theta) * np.cos(phi),
                          r * np.cos(theta) * np.sin(phi),
                          r * np.sin(theta))
    def r(self):
        '''return r, which is the norm of the vector'''
        return self.norm()
    
    def phi(self):
        '''return phi, which is found from the x and y components'''
        return np.arctan2(self.y, self.x)
    
    def theta(self):
        '''return theta, which is found from z and r'''
        return np.arcsin(self.z / self.r())


# In[ ]:


a = MyVector(x=3., y=1., z=0.)
b = MyPolarVector(r=1., theta=np.pi/4., phi=0.)
print(f'{a}, {b}')
print(f'b: r = {b.r():.5f}, theta = {b.theta():.5f}, phi = {b.phi():.5f}')


# In[ ]:




