# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 15:29:15 2022

@author: mcdon
"""


"""
This code implements a score based sampler for a one dimensional multimodal 
density. The density is defined by p(x) = exp(f(x)-0.5 x^2). To run the score 
sampler, the score must be estimated using a monte carlo approach to 
empirically estimate the score at each point in question. two different estimators
are used at different time scales. Operations are vectorized where possible to
improve run time.
"""

#Initial Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bisect
import scipy.stats as st
import time



#Initialize Variables
#parameters of log likelihood function f(x)
centers = np.array([-5, -1, 3,4])
scale_value = 100
a = 1
b = 0.05
inc = 0.01
num_samples = 1000000

#function which defines log likelihood
def f_func(x):
    val = scale_value*np.sum([np.tanh(a*x+b-u)-np.tanh(a*x-b-u) for u in centers])
    return val

#gradient of log likelihood
def f_grad_func(x):
    val = scale_value*np.sum([a*np.cosh(a*x+b-u)**(-2)-a*np.cosh(a*x-b-u)**(-2)for u in centers])
    return val

#plot f function for visualization

test_vals = np.arange(-10, 10, step = inc)
f_vals = [f_func(v) for v in test_vals]

plt.figure()
plt.plot(test_vals, f_vals)

#Pdf values for later plotting

y_vals = np.exp([f_func(v)-0.5*(v**2) for v in test_vals])
pdf_vals = y_vals/np.sum(y_vals*inc)


#---------------------Vectorized MCMC Part---------------------------------------------------

#in order for code to run quickly, operations are vectorized with base
#numpy operations to improve evaluation time

#efficient function to evaluate f at multiple points in an array at once
def fast_vectorized_f(x):
    #x is now an array of elements to be evaluated
    input_matrix = np.array([a*x-c for c in centers])
    vals_1 = np.tanh(input_matrix+b)
    vals_2 = np.tanh(input_matrix-b)
    output = vals_1 - vals_2
    output = np.sum(output, axis = 0)
    return scale_value * output

#efficient function to evaluate grad f at multiple points in an array at once
def fast_vectorized_grad_f(x):
    #x is now an array of elements to be evaluated
    input_matrix = np.array([a*x-c for c in centers])
    vals_1 = np.cosh(input_matrix+b)**-2
    vals_2 = np.cosh(input_matrix-b)**-2
    output = a*vals_1 - a*vals_2
    output = np.sum(output, axis = 0)
    return scale_value * output


#the score can be estimated in one of two ways using integration by part. 
#the function is passed an array of values and computes score estimate at all 
#given values. For each value, K normal samples are generated, and the empirical
#average computed for each of the input values.
def fast_vectorized_est_score_v3(thetas, t, K = 1000):
    #thetas is array of input values, t is time, K is number of samples per 
    #empirical estimate
    
    #random noise values
    U = np.random.normal(size = (K, np.shape(thetas)[0]))
    
    sigma_t = np.sqrt(1-np.exp(-2*t)) #time varying sd
    input_vals = np.exp(-t)*thetas + sigma_t*U #evaluation points
    w = np.exp(fast_vectorized_f(input_vals)) #compute weights
    colsums = np.sum(w, axis = 0)
    w = w/colsums #normalize to sum to 1
    scores = -thetas+(np.exp(-t)/sigma_t)*np.sum(U*w, axis = 0) #return scores
    return scores



def fast_vectorized_est_score_v4(thetas, t, K = 1000):
    #thetas is array of input values, t is time, K is number of samples per 
    #empirical estimate
    
    #random noise values
    U = np.random.normal(size = (K, np.shape(thetas)[0]))
    #compute weights
    sigma_t = np.sqrt(1-np.exp(-2*t)) #time varying sd
    input_vals = np.exp(-t)*thetas + sigma_t*U #evaluation points
    w = np.exp(fast_vectorized_f(input_vals)) #compute weights
    colsums = np.sum(w, axis = 0)
    w = w/colsums #normalize to sum to 1
    Vals = fast_vectorized_grad_f(input_vals)
    scores = -thetas+np.exp(-t)*np.sum(Vals*w, axis = 0) #return scores
    return scores

#-----------------------------Vectorized Sampling----------------------------------------

delta = 0.01 #Step Size
num_empirical_samples = 1000 #number of samples of disitribution to return
samples_per_estimate = 3000 #K value in Monte Carlo score estimator

initial_samples = np.random.normal(size = num_empirical_samples)

t_sequence = np.arange(delta, 2, step = delta)[::-1]

#matrix to store samples at each time step for later plotting
STORE_samples = np.ndarray(shape = [np.shape(t_sequence)[0], num_empirical_samples])

current_samples = initial_samples

#implement backwards flow
for i in np.arange(0, STORE_samples.shape[0], step = 1):
    if t_sequence[i] > 0.1:
        scores = fast_vectorized_est_score_v3(current_samples, t_sequence[i],
                                              K = samples_per_estimate)
    else:
        scores = fast_vectorized_est_score_v4(current_samples, t_sequence[i], 
                                             K = samples_per_estimate)
    current_samples = current_samples - delta*(-current_samples-2*scores)+np.sqrt(2*delta)*np.random.normal(size = num_empirical_samples)
    STORE_samples[i,:] = current_samples
    print((i+1)/(STORE_samples.shape[0]+1))

#histogram of initial samples
plt.figure()
plt.hist(initial_samples, bins = 30, density = True)

#histogram of final samples
fig = plt.figure()
plt.xlim(-5,5)
plt.hist(current_samples, bins = 50, density = True, color = "orange")
plt.plot(test_vals, pdf_vals)

#fig.savefig("basic_one_dim.jpg", format = "jpg", dpi = 1200)

#evolving histogram over time
for i in np.arange(0, STORE_samples.shape[0], step = 1):
    plt.figure()
    plt.hist(STORE_samples[i,:], bins = 50,     density = True)
    plt.plot(test_vals, pdf_vals)
    plt.xlim([-5,5])
    plt.ylim([0,1])
                