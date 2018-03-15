# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 18:50:51 2018

@author: emberiza
"""
import numpy as np
# note the quadgk.py file has to be in your python path
from quadgk import quadgk

#Integrate f(z) = 1/(2z-1) in the complex plane over the triangular
def f1(z):
    return 1/(2*z-1)
Q = quadgk(f1,0,0,Waypoints=np.array([1+1j,1-1j]))
print(Q[0])

#   Integrate f(x) = exp(-x^2)*log(x)^2 from 0 to infinity:
def f2(x):
    return np.exp(-x**2)*np.log(x)**2
Q = quadgk(f2,0,np.Inf)
print(Q[0],Q[1])

def f4(x,c):
    return np.exp(-c*x**2)
c=np.pi
Q = quadgk(f4,-np.Inf,np.Inf,c)
print(Q[0],Q[1])
