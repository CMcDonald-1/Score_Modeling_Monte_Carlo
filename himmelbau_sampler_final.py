# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 12:12:00 2022

@author: mcdon
"""

"""
This code implements a score based sampler for a known multi-modal density in 2
dimensions. The density is p(x,y) = exp(f(x,y)-0.5(x^2+y^2)) where f(x,y) is the 
Himmelblau function f(x,y) = -(x^2+y-11)^2 - (x+y^2-7)^2. This has modes at 
 (3,2), (-2.81, 3.13), (-3.78, -3.28), (3.58, -1.85). The score is estimated via
a Monte Carlo empirical average using two different estimators at different time 
scales. Where possible, operations are vectorized and base numpy operators used
to speed up evaluation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bisect
import scipy.stats as st
import time
from mpl_toolkits import mplot3d
import os

#Himmelblau function f(x,y)
def himmelbau(x):
    return(-((x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2))



#vectorized function to estimate score. It is passed a matrix of n by 2 values 
#theat, and for eacy K by 2 normal values are generated and the score averaged over
#these values for each input row of theta.
def est_scores_v3(thetas,t, K = 1000):
    N = np.shape(thetas)[0]
    #3 dimensional tensor for vectorized operations
    U = np.random.normal(size = (N,2,K))
    
    sigma_t = np.sqrt(1-np.exp(-2*t)) #time varying sd
    
    input_vals = np.exp(-t)*thetas[:, :,np.newaxis]+sigma_t*U #3 dim tensor to average over

    #vectorized evaluation of Himmelblau function at input tensor 
    h_vals = -(input_vals[:,0,:]**2+input_vals[:,1,:]-11)**2-(input_vals[:,0,:]+input_vals[:,1,:]**2-7)**2
    h_vals = h_vals + 0.5*(input_vals[:,0,:]**2+input_vals[:,1,:]**2)
    w = np.exp(h_vals)
    w = w/np.sum(w, 1)[:, np.newaxis]

    #sum over correct dimensions of tensor to produce scores
    scores = -thetas+ (np.exp(-t)/sigma_t)*np.sum(w[:, np.newaxis, :]*U, axis = 2)
    return scores


def est_scores_v4(thetas, t, K = 1000):
    N = np.shape(thetas)[0]
    #3 dimensional tensor for vectorized operations
    U = np.random.normal(size = (N,2,K))
    
    sigma_t = np.sqrt(1-np.exp(-2*t)) #time varying sd
    
    input_vals = np.exp(-t)*thetas[:, :,np.newaxis]+sigma_t*U #3 dim tensor to average over
    
    #vectorized evaluation of Himmelblau function at input tensor 
    h_vals = -(input_vals[:,0,:]**2+input_vals[:,1,:]-11)**2-(input_vals[:,0,:]+input_vals[:,1,:]**2-7)**2
    h_vals = h_vals + 0.5*(input_vals[:,0,:]**2+input_vals[:,1,:]**2)
    w = np.exp(h_vals)
    w = w/np.sum(w, 1)[:, np.newaxis]
    
    #compute gradients
    x = input_vals[:,0,:]
    y = input_vals[:,1,:]
    
    #vectorized evaluation of gradient at Himmelblau gradient function
    grad_vals = [-4*(x**2+y-11)*x-2*(x+y**2-7), -2*(x**2+y-11)-4*(x+y**2-7)*y]
    scores = -thetas+np.exp(-t)*np.transpose((np.sum(grad_vals*w[np.newaxis, :,:], axis = 2)))
    return scores
    


#----------------------------------MCMC Sampling-------------------------------------
delta = 0.01 #step size
num_empirical_samples = 2000 #number of samples of disitribution to return
samples_per_estimate = 1000 #K value in Monte Carlo score estimator
T_final = 3     #Terminal time

initial_samples = np.random.normal(size = (num_empirical_samples, 2))
t_sequence = np.arange(delta, T_final, step = delta)[::-1]

#tensor to store samples at each time stage at index [:,:,i]
STORE_samples = np.ndarray(shape = (num_empirical_samples, 2, np.shape(t_sequence)[0]))

current_samples = initial_samples


for i in np.arange(0, np.shape(t_sequence)[0], step = 1):
    if(t_sequence[i]> 0.1):
        scores = est_scores_v3(current_samples, t_sequence[i], K = samples_per_estimate)
    else:
        scores = est_scores_v4(current_samples, t_sequence[i], K = samples_per_estimate)
        
    current_samples = current_samples - delta*(-current_samples-2*scores)+np.sqrt(2*delta)*np.random.normal(size = (num_empirical_samples, 2))
    STORE_samples[:,:,i] = current_samples
    print(np.round((i+1)/(np.shape(t_sequence)[0]+1),4))
    
#countour plot 
xlist = np.linspace(-5.0, 5.0, 100)
ylist = np.linspace(-5.0, 5.0, 100)
X, Y = np.meshgrid(xlist, ylist)

Z = np.ndarray(shape = (100,100))

for i in np.arange(0, 100):
    for j in np.arange(0,100):
        Z[i,j] = himmelbau([X[i,j], Y[i,j]])
        Z[i,j] = Z[i,j]-0.5*(X[i,j]**2+Y[i,j]**2)

fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z, 30, cmap='inferno')
fig.colorbar(cp) # Add a colorbar to a plot
cp = ax.contour(X, Y, Z, 100, colors = "black", levels = [-100, -50, -25, -10] )
ax.clabel(cp, inline=True, fontsize=8)
plt.scatter(current_samples[:,0], current_samples[:,1])

#fig.savefig("himmelbau_heatmap.jpg", format = "jpg", dpi = 1200)


#compare relative proportions of points around each mode to estimate of integral
#of pdf around each mode
epsilon_width = 0.5 #square of +/- epsilon in each coordinate around each mode

mode_1 = np.array([3,2])
shifts = np.random.uniform(size = (2, 10000),low = -epsilon_width, high = epsilon_width)
temp_vals = mode_1[:, np.newaxis]+shifts
p_1 = np.mean(np.exp(np.apply_along_axis(himmelbau, 0, temp_vals)- 0.5*temp_vals[0,:]**2-0.5*temp_vals[1,:]**2))

mode_2 = np.array([-2.81, 3.13])
shifts = np.random.uniform(size = (2, 10000),low = -0.5, high = 0.5)
temp_vals = mode_2[:, np.newaxis]+shifts
p_2 = np.mean(np.exp(np.apply_along_axis(himmelbau, 0, temp_vals)- 0.5*temp_vals[0,:]**2-0.5*temp_vals[1,:]**2))

mode_3 = np.array([-3.78, -3.28])
shifts = np.random.uniform(size = (2, 10000),low = -epsilon_width, high = epsilon_width)
temp_vals = mode_3[:, np.newaxis]+shifts
p_3 = np.mean(np.exp(np.apply_along_axis(himmelbau, 0, temp_vals)- 0.5*temp_vals[0,:]**2-0.5*temp_vals[1,:]**2))

mode_4 = np.array([3.58, -1.85])
shifts = np.random.uniform(size = (2, 10000),low = -epsilon_width, high = epsilon_width)
temp_vals = mode_4[:, np.newaxis]+shifts
p_4 = np.mean(np.exp(np.apply_along_axis(himmelbau, 0, temp_vals)- 0.5*temp_vals[0,:]**2-0.5*temp_vals[1,:]**2))

p_sum = p_1+p_2+p_3+p_4

#how many sampled points are near each mode

m1_points = np.mean(np.max(np.abs(current_samples - mode_1[np.newaxis, :]), axis = 1)< epsilon_width)

m2_points = np.mean(np.max(np.abs(current_samples - mode_2[np.newaxis, :]), axis = 1)< epsilon_width)

m3_points = np.mean(np.max(np.abs(current_samples - mode_3[np.newaxis, :]), axis = 1)< epsilon_width)

m4_points = np.mean(np.max(np.abs(current_samples - mode_4[np.newaxis, :]), axis = 1)< epsilon_width)

m_sum = m1_points+m2_points+m3_points+m4_points


#probability of each mode region
[p_1, p_2, p_3, p_4]/p_sum

#sampled proportions of each mode region
[m1_points, m2_points, m3_points, m4_points]/m_sum

fig = plt.figure()
plt.scatter([p_1, p_2, p_3, p_4]/p_sum, [m1_points, m2_points, m3_points, m4_points]/m_sum)
plt.xlabel("Probability of Each Mode")
plt.ylabel("Sample Proportion of Each Mode")
plt.title("Expected vs Actual Sampling Proportions of Each Mode")
plt.plot([0,1], [0,1], ls = "--")

