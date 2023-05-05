# ML-Final-Project
Supporting Code for my ECE50024 PDF submission

pickle files containing my curated testing and training images and ID's are provided 
please be sure that ALL 4 FILES are in the file path so my script can import the necessary data

I avoided using libraries like GPyOpt and BayesOpt in order to have a deeper understanding of the code
However, this introduces some instability

Due to version incompatibility between and sklearn.gaussian_process and scipy, this code will likely not run 
with the lastest versions. I am running this script on scipy version 1.9.1 and sklearn 1.1.3

Even after changing versions, there are still compatibility issues. For this reason, in lines 298-321, I create 
a custom instantiation of GaussianProcessRegressor but modify the 'max_iter' and 'method' items.

These edits cause the program to print "RuntimeWarning: Values in x were outside bounds during a minimize step, clipping to bounds" 
on most iterations. This is expected and does not affect performance. Using this more flexible solver is the only way I could get 
things to work. 

################################
ATTENTION
################################

If the above issues cause my script to fail to run on your computer, please consider changing "MyGPR" on line 341 to "GaussianProcessRegressor"
