import torchtt as tntt
import numpy as np
import tt_iga
import matplotlib.pyplot as plt
import torch as tn

tn.set_default_dtype(tn.float64)

def create_geometry( ):
    
    Nt = 24                                                                
    lz = 40e-3                                                             
    Do = 72e-3                                                            
    Di = 51e-3                                                            
    hi = 13e-3                                                             
    bli = 3e-3                                                             
    Dc = 3.27640e-2                                                           
    hc = 7.55176e-3                                                           
    ri = 20e-3                                                           
    ra = 18e-3                                                           
    blc = hi-hc                                                           
    rm = (Dc*Dc+hc*hc-ri*ri)/(Dc*np.sqrt(2)+hc*np.sqrt(2)-2*ri)                 
    R = rm-ri

    get_rO = lambda Ax,Ay,ri: (Ax**2+Ay**2-ri**2)/(np.sqrt(2)*(Ax+Ay)-2*ri)

    O = np.array([rm/np.sqrt(2),rm/np.sqrt(2)])
    alpha1 = -np.pi*3/4       
    alpha2 = np.math.asin((hc-rm/np.sqrt(2))/R)
    alpha = np.abs(alpha2-alpha1)
    
    A = np.array([[O[0] - ri/np.sqrt(2), O[1] - ri/np.sqrt(2)], [O[0] - Dc, O[1] - hc]])
    b = np.array([[A[0,0]*ri/np.sqrt(2)+A[0,1]*ri/np.sqrt(2)],[A[1,0]*Dc+A[1,1]*hc]])
    C = np.linalg.solve(A,b)

    control_points = tt_iga.geometry.ParameterDependentControlPoints([7,5])

    control_points[:,0,0] = [0,0]
    control_points[:,1,0] = [lambda params: (Dc+params[1])/2, 0]
    control_points[:,2,0] = [lambda params: (Dc+params[1]), 0]
    control_points[:,3,0] = [lambda params: (Dc+params[1]+Di)/2,0]
    control_points[:,4,0] = [Di,0]
    control_points[:,4,0] = [Do,0]

    
    knots = np.zeros((5,5,2))   
     
    knots[1,0,:] = np.array([Dc/2,0])
    knots[2,0,:] = np.array([Dc,0])
    knots[3,0,:] = np.array([Di,0])
    knots[4,0,:] = np.array([Do,0])

    knots[0,2,:] = np.array([ri/np.sqrt(2),ri/np.sqrt(2)])
    knots[1,2,:] = np.array([C[0,0],C[1,0]])
    knots[2,2,:] = np.array([Dc,hc])
    knots[3,2,:] = np.array([Di,hi-bli])
    knots[4,2,:] = np.array([Do,hi-bli])

    knots[:,1,:] = 0.5*(knots[:,0,:]+knots[:,2,:])
    
    knots[0,3,:] = np.array([(0.75*ri+0.25*Do)/np.sqrt(2),(0.75*ri+0.25*Do)/np.sqrt(2)])
    knots[2,3,:] = np.array([Dc+blc,hi])
    knots[1,3,:] = 0.5*(knots[0,3,:]+knots[2,3,:])
    knots[3,3,:] = np.array([Di-bli,hi])
    knots[4,3,:] = np.array([Do,hi])
    
    knots[4,4,:] = np.array([Do,Do*np.tan(np.pi/8)])
    knots[0,4,:] = np.array([Do/np.sqrt(2),Do/np.sqrt(2)])
    knots[1,4,:] = 0.75*knots[0,4,:]+0.25*knots[4,4,:]
    knots[2,4,:] = 0.5*knots[0,4,:]+0.5*knots[4,4,:]
    knots[3,4,:] = 0.25*knots[0,4,:]+0.75*knots[4,4,:]

    knots_new = np.zeros((7,5,2))
    knots_new[0,...] = knots[0,...]
    knots_new[1,...] = knots[1,...]
    knots_new[2,...] = knots[2,...]
    knots_new[3,...] = 0.5*(knots[2,...]+knots[3,...])
    knots_new[4,...] = knots[3,...]
    knots_new[5,...] = 0.5*(knots[3,...]+knots[4,...])
    knots_new[6,...] = knots[4,...]
    
    weights = np.ones(knots_new.shape[:2])
    weights[1,2] = np.sin((np.pi-alpha)/2)
    
    return knots_new, weights
