# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 12:23:02 2014

@author: 071cht
"""
"""
NOTE: 
FANN_SIGMOID: y= 0 or 1
FANN_SIGMOID_SYMMERIC y=-1 or 1 

"""
#!/usr/bin/python
from pyfann import libfann
import numpy as np

connection_rate=1
learning_rate=0.7
num_input=35
num_neurons_hidden = (35+1)*1/2
num_output = 1

desired_error =  0.0000000001
max_iterations= 100000
iterations_between_reports = 100

#import data 
train_name='cv_train2.train'
test_name='cv_test2.test'

train_data = libfann.training_data()
train_data.read_train_from_file(train_name)
test_data = libfann.training_data()
test_data.read_train_from_file(test_name)

path='/Users/071cht/Desktop/chase/sample/'
train_m = np.loadtxt(path+'cv_train2.txt',delimiter= ',')
test_m= np.loadtxt(path+'cv_test2.txt',delimiter= ',')
feature=test_m[:,0:-1]
label=test_m[:,-1]

#set up neuro network
ann = libfann.neural_net()
ann.create_sparse_array(connection_rate, (num_input, num_neurons_hidden, num_output))
ann.set_learning_rate(learning_rate)
ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC)
ann.set_train_error_function(libfann.ERRORFUNC_TANH)
ann.train_on_data(train_data, max_iterations, iterations_between_reports, desired_error);

#ann.print_parameters();
#ann.print_connections();

print "\nTrain error: %f, Test MSE: %f\n\n" %( ann.test_data(train_data),ann.test_data(test_data))

# test classification accuracy
out=list()
for i in range(feature.shape[0]):
    out_elm=ann.run(feature[i])
    out.append(out_elm)
out_array=np.asarray(out)
pred=np.where(out_array<0,0,out_array)
pred=np.where(out_array>=0,1,pred)

label=label.reshape(len(label),1)
true=np.where(label<0,0,label)
true=np.where(label>=0,1,true)

error_elm=sum(np.abs(pred-true))/len(label)
Acc_elm=1-error_elm
print Acc_elm
#ann.save("nets/xor_float.net");
