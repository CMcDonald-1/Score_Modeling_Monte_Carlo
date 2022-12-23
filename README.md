# Score_Modeling_Monte_Carlo
Implementation of a score based approach to sampling using Monte Carlo estimation of score on one and two dimensional examples. Companion code for NeurIPS 2022 workshop paper.

This code is the companion code for the paper "Proposal of a Score Based Approach to Sampling Using Monte Carlo Estimation of Score and Oracle Access to Target Density" at the Score Based Methods Workshop at NeurIPS 2022. In this code, we implement the score based sampler by estimating the score using empirical averages of a Monte Carlo estimator. The method is demonstrated on a multi-modal density in 1 dimensions and the Himmelblau function in 2 dimensions as the log liklihood. The algorithms are coded in python with vectorization and base numpy functions used when possible to speed up runtime.

Files:
* ***basic_neurips_example_final.py***: implementation of algorithm on 1 dimensional example
* ***himmelblau_sampler_final.py***: implementation of algorithm on 2 dimensional example
* ***test_animation.mp4***: animation of 1 dimensional histogram example as samples converge to target density
