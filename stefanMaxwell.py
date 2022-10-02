#   --- STEFAN, MAXWELL DIFFUSION MODEL ---

#   Technische Universiteit Delft - TUDelft (2022)
#
#   Master of Science in Chemical Engineering
#
#   This code was developed and tested by Elia Ferretti
#
#   You can redistribute the code and/or modify it
#   Whenever the code is used to produce any publication or document,
#   reference to this work and author should be reported
#   No warranty of fitness for a particular purpose is offered
#   The user must assume the entire risk of using this code

#   ---------------------------------------------------------------------------------------------------------

def matrixD(n,diff,x):
# n = number of diffusive species
# diff = matrix of diffusivities
# x = vector of the species' molar fraction
    B = np.zeros((n-1,n-1))
    gamma = np.eye(n-1)
    for i in range(0,n-1):
        for j in range(0,n-1):
            if i==j:
                for k in range(0,n):
                    sum = 0
                    if k!=i:
                        sum += x[k]/diff[i,k]
                B[i,j] = x[i]/diff[i,n-1] + sum
            else:
                B[i,j] = - x[i]*(1/diff[i,j] - 1/diff[i,n-1])
    if np.any(B):            
        D = np.dot(np.linalg.inv(B),gamma)
    else:
        D = np.zeros((n-1,n-1))   
    return D

def diffusiveTerm(n,position,diff,c,h):
# n = number of diffusive species
# position = index of point where to compute the diffusive term
# diff = matrix of diffusivities
# c = concentration field
# h = space discretization

    D_central = matrixD(n,diff,c[:,position])
    D_forward = matrixD(n,diff,c[:,position+1])
    D_back    = matrixD(n,diff,c[:,position-1])
    
    #centered differencing scheme for first derivatives
    dDdz = (D_forward-D_back)/(2*h)
    dcdz = (c[0:n-1,position+1]-c[0:n-1,position-1])/(2*h)
    
    #centered differencing scheme for second derivative
    d2cdz2 = (c[0:n-1,position+1]-2*c[0:n-1,position]+c[0:n-1,position-1])/h**2
    
    #reshape vector into column vectors
    dcdz = np.reshape(dcdz,(n-1,1))
    d2cdz2 = np.reshape(d2cdz2,(n-1,1))
    
    N = np.dot(dDdz,dcdz) + np.dot(D_central,d2cdz2)
    
    return  N

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import time

#----------------------------------------------------------------------------------------------------------------

# 1 - Acetone
# 2 - Methanol
# 3 - Air
n = 3

diffusivities = np.array([[0,8.48e-6,13.72e-6],[8.48e-6,0,19.91e-6],[13.72e-6,19.91e-6,0]])

#domain
l = 0.2425
gridPoints = 100
x = np.linspace(0,l,gridPoints)
h = x[1]-x[0]
totTime = 2000
t = 0
dt = 0.05   
timeIteration = 0

#graphical post-processing
graphStep = 1e3
onTheFlyPP = True

#pre-allocation and initial condition
c = np.zeros((3,gridPoints))
c[n-1,:] = 1;

#boundary condition
c[0,0] = 0.3173
c[1,0] = 0.5601
c[2,0] = 0.1227

while t<totTime:
        
    for r in range(1,gridPoints-1):
        #grid point for loop        
        diffusion = diffusiveTerm(n,r,diffusivities,c,h)
        
        for s in range(0,n-1):
            #species for loop (exept n-th species)
            c[s,r] = c[s,r] + dt*diffusion[s];
    
    #using stoichiometric equation to determine n-th species molar fraction
    c[2,:] = 1       
    for k in range(0,n-1):        
        c[2,:] -= c[k,:]
    
    #advance in time
    timeIteration += 1
    t += dt
    
    #onTheFlyPP
    if onTheFlyPP and timeIteration%graphStep==0:
        plt.plot(x/l,c[0,:],x/l,c[1,:],x/l,c[2,:])
        title = "Profile at t = " + str(round(t,2)) + " [s]"
        plt.title(title)
        plt.ylabel("x [-] (molar fractions)")
        plt.xlabel("z [-] (adimensional tube length)")
        plt.legend(["Acetone","Methanol","Air"])
        plt.show()
        time.sleep(0.01)
        