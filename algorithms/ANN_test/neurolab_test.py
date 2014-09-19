# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:01:17 2014

@author: 071cht
"""

""" 
Example of use multi-layer perceptron
=====================================

Task: Approximation function: 1/2 * sin(x)

"""

import neurolab as nl
import numpy as np

# Create train samples
x = np.linspace(-7, 7, 20)
#x[0]=np.nan
#x[5]=np.inf

y = np.sin(x) * 0.5

size = len(x)

inp = x.reshape(size,1) #transpose x
tar = y.reshape(size,1)

# Create network with 2 layers and random initialized
#[-7,7] is range of input
#[5,1] means 5 neurons in first hidden layer, 1 neurons in output layer 

net = nl.net.newff([[-10, 10]],[5, 1])

# Train network
error = net.train(inp, tar, epochs=500, show=100, goal=0.01)

print inp.shape, tar.shape, net.ci,net.co

# Simulate network
out = net.sim(inp)

print

# Plot result
import pylab as pl
pl.subplot(211)
pl.plot(error)
pl.xlabel('Epoch number')
pl.ylabel('error (default SSE)')

x2 = np.linspace(-6.0,6.0,150)
y2 = net.sim(x2.reshape(x2.size,1)).reshape(x2.size)
print y2.shape

y3 = out.reshape(size)
print y3.shape

pl.subplot(212)
pl.plot(x2, y2, '-',x , y, '.', x, y3, '*')
pl.legend(['train target', 'net output'])
pl.show()