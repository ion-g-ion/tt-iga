import torch as tn
import torchtt as tntt
import matplotlib.pyplot as plt
import tt_iga
import numpy as np
import datetime
import scipy.sparse
import scipy.sparse.linalg
import pandas as pd

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
    O = np.array([rm/np.sqrt(2),rm/np.sqrt(2)])
    alpha1 = -np.pi*3/4       
    alpha2 = np.math.asin((hc-rm/np.sqrt(2))/R)
    alpha = np.abs(alpha2-alpha1)
    
    A = np.array([[O[0] - ri/np.sqrt(2), O[1] - ri/np.sqrt(2)], [O[0] - Dc, O[1] - hc]])
    b = np.array([[A[0,0]*ri/np.sqrt(2)+A[0,1]*ri/np.sqrt(2)],[A[1,0]*Dc+A[1,1]*hc]])
    C = np.linalg.solve(A,b)

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

knots, weights = create_geometry()

plt.figure()
plt.scatter(knots[:,:,0],knots[:,:,1],s=2)

basis1 = tt_iga.bspline.BSplineBasis(np.array([0,0.4,0.4,0.6,0.6,1]),2)
basis2 = tt_iga.bspline.BSplineBasis(np.array([0,0.15,0.3,0.5,1]),1)

geom = tt_iga.PatchNURBS([basis1, basis2],[], [tntt.TT(knots[:,:,0]), tntt.TT(knots[:,:,1])], tntt.TT(weights))

y1, y2 = np.linspace(0.4,0.6,64), np.linspace(0,0.5,64)
X1,X2 = geom([y1,y2])
plt.figure()
plt.scatter(X1.numpy().flatten(), X2.numpy().flatten(),s=1,c='orange')

y1, y2 = np.linspace(0,0.4,64), np.linspace(0,0.3,64)
X1,X2 = geom([y1,y2])
plt.scatter(X1.numpy().flatten(), X2.numpy().flatten(),s=1,c='blue')

y1, y2 = np.linspace(0.6,1,64), np.linspace(0,0.5,64)
X1,X2 = geom([y1,y2])
plt.scatter(X1.numpy().flatten(), X2.numpy().flatten(),s=1,c='red')

y1, y2 = np.linspace(0,1,128), np.linspace(0.5,1,128)
X1,X2 = geom([y1,y2])
plt.scatter(X1.numpy().flatten(), X2.numpy().flatten(),s=1,c='red')

y1, y2 = np.linspace(0,0.4,64), np.linspace(0.3,0.5,64)
X1,X2 = geom([y1,y2])
plt.scatter(X1.numpy().flatten(), X2.numpy().flatten(),s=1,c='red')

y1, y2 = np.linspace(0,1,128), np.linspace(0.,1,128)
X1,X2 = geom([y1,y2])
plt.figure()
plt.scatter(X1.numpy().flatten(), X2.numpy().flatten(),s=1,c='green')


mu0 = 4*np.pi*1e-7
mur = 1000
mu_ref = lambda y: 1/mu0*((y[...,1]<0.5)*(y[...,0]<0.6)*(y[...,0]>0.4)+(y[...,1]<0.3)*(y[...,0]<0.4))+1/(mu0*mur)*tn.logical_not((y[...,1]<0.5)*(y[...,0]<0.6)*(y[...,0]>0.4)+(y[...,1]<0.3)*(y[...,0]<0.4))

basis_solution = [tt_iga.bspline.BSplineBasis(np.concatenate((np.linspace(0,0.4,32), np.linspace(0.4,0.6,16),np.linspace(0.6,1,32))),2)]
basis_solution.append(tt_iga.bspline.BSplineBasis(np.concatenate((np.linspace(0,0.15,20),np.linspace(0.15,0.3,20), np.linspace(0.3,0.5,20),np.linspace(0.5,1,20))),2))
Mass_tt = geom.mass_interp(basis_solution)
Stiff_tt = geom.stiffness_interp(basis_solution, func_reference=mu_ref)

Jref = lambda y: 1000000*(y[...,1]<0.5)*(y[...,0]<0.6)*(y[...,0]>0.4)+0.0

rhs_tt = geom.rhs_interp(basis_solution,Jref)

P1 = tn.eye(Mass_tt.N[0])
P2 = tn.eye(Mass_tt.N[0])
P2[-1,-1] = 0
P1[0,0] = 0
P1[-1,-1] = 0
Pin_tt = tntt.rank1TT([P1,P2])
Pbd_tt = tntt.eye(Mass_tt.N) - Pin_tt

M_tt = (Pin_tt@Stiff_tt+Pbd_tt).round(1e-12)
rhs_tt = (Pin_tt @ rhs_tt + 0).round(1e-12)

print('System matrix... ',flush=True)


print('Rank Mtt ',M_tt.R)
print('Rank rhstt ',rhs_tt.R)

tme = datetime.datetime.now() 
# dofs_tt = tntt.solvers.amen_solve(M_tt.cuda(), rhs_tt.cuda(), x0 = tntt.ones(rhs_tt.N).cuda(), eps = eps_solver, nswp = 50, kickrank = 4, preconditioner = 'c', verbose = False).cpu()
dofs_tt = tntt.solvers.amen_solve(M_tt, rhs_tt, x0 = tntt.ones(rhs_tt.N), eps = 1e-8, nswp = 60, kickrank = 4, preconditioner = 'c', verbose = True)
tme = datetime.datetime.now() - tme
print('Time system solve ',tme,flush=True)



plt.figure()
y1, y2 = np.linspace(0,1,201), np.linspace(0.,1,201)
X1,X2 = geom([y1,y2])
u = dofs_tt.mprod([tn.tensor(basis_solution[0](y1).T),tn.tensor(basis_solution[1](y2).T)],[0,1])
plt.contourf(X1.numpy(), X2.numpy(),u.numpy(),levels=32)
plt.colorbar()


plt.figure()
y1, y2 = np.linspace(0,1,201), np.linspace(0.,1,201)
X1,X2 = geom([y1,y2])
u = dofs_tt.mprod([tn.tensor(basis_solution[0](y1).T),tn.tensor(basis_solution[1](y2).T)],[0,1])
plt.contour(X1.numpy(), X2.numpy(),u.numpy(),levels=32)
plt.colorbar()