import torch as tn
import torchtt as tntt
import numpy as np
from ._aux_functions import *
import matplotlib.pyplot as plt
import datetime


class Function():
      
    def __init__(self, basis):
        """
        

        Args:
            basis ([type]): [description]
        """
        self.N = [b.N for b in basis]
        self.basis = basis
        
    def interpolate(self, function, geometry = None, eps = 1e-12):
        """
        

        Args:
            function ([type]): [description]
            geometry ([type], optional): [description]. Defaults to None.
            eps ([type], optional): [description]. Defaults to 1e-12.

        Returns:
            [type]: [description]
        """
        Xg = [tn.tensor(b.interpolating_points()[0], dtype = tn.float64) for b in self.basis]
        Mg = [tn.tensor(b.interpolating_points()[1], dtype = tn.float64) for b in self.basis]
        
        corz = [tn.reshape(Mg[i].t(), [1,Mg[i].shape[0],-1,1]) for i in range(len(Mg))]
        Gm = tntt.TT(corz)
        
        if geometry == None:
            X = tntt.TT(Xg[0])**tntt.ones(self.N[1:])   
            Y = tntt.ones(self.N[:1]) ** tntt.TT(Xg[1]) ** tntt.ones(self.N[2:]) 
            Z = tntt.ones(self.N[:2]) ** tntt.TT(Xg[2]) ** tntt.ones(self.N[3:])  
        else:
            X,Y,Z = geometry(Xg)

        if len(self.basis)==3:
            evals = tntt.interpolate.function_interpolate(function, [X, Y, Z], eps)
        else:
            Np = len(self.basis[3:])
            meshgrid = tntt.meshgrid([x for x in Xg[3:]])
            meshgrid = [X,Y,Z] + [tntt.ones([n for n in self.N[:3]])**m for m in meshgrid]
            evals = tntt.interpolate.function_interpolate(function, meshgrid, eps, verbose = False)
            

        dofs = tntt.solvers.amen_solve(Gm,evals,x0 = evals,eps = eps,verbose = False)
        self.dofs = dofs
        
        return dofs
    
    def __call__(self, x, deriv = None):
        """
        

        Args:
            x ([type]): [description]
            deriv ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if isinstance(x,list):
            if deriv == None:
                deriv = [False]*len(self.N)
            
            Bs = [tn.tensor(self.basis[i](x[i].numpy(),derivative=deriv[i]), dtype = tn.float64).t() for i in range(len(self.N))]
            B_tt = tntt.TT([tn.reshape(m,[1,m.shape[0],-1,1]) for m in Bs])
            
            val = B_tt @ self.dofs
        else:
            if deriv == None:
                deriv = [False]*len(self.N)
            
            Bs = [tn.tensor(self.basis[i](x[:,i].numpy(),derivative=deriv[i]), dtype = tn.float64) for i in range(len(self.N))]

            val = tn.ones((x.shape[0],1))

            for i in range(len(self.N)):
                tmp = tn.tensordot(val,self.dofs.cores[i],([1],[0]))
                val = tn.einsum('ikl,ki->il', tmp,Bs[i])
            val = val[:,0]

        return val
    
    def L2error(self, function, geometry_map = None, level = 32):
        """
        

        Args:
            function ([type]): [description]
            geometry_map ([type], optional): [description]. Defaults to None.
            level (int, optional): [description]. Defaults to 32.

        Returns:
            [type]: [description]
        """
        pts, ws = np.polynomial.legendre.leggauss(level)
        pts = (pts+1)*0.5
        ws = ws/2
        
        Xg = [tn.tensor(b.interpolating_points()[0], dtype = tn.float64) for b in self.basis]
         
        if geometry_map != None:
            X,Y,Z = geometry_map([tn.tensor(pts)]*len(self.N))
            Og_tt = geometry_map.eval_omega([tn.tensor(pts)]*3)
        else:
            X,Y,Z = geometry_map(Xg)
        
        B_tt = tntt.TT(self.basis[0](pts),shape = [(self.basis[0].N,pts.size)]) ** tntt.TT(self.basis[1](pts),shape = [(self.basis[1].N,pts.size)]) ** tntt.TT(self.basis[2](pts),shape = [(self.basis[2].N,pts.size)]) 
        B_tt = B_tt.t()
        C_tt = tntt.eye(B_tt.M)
        
        for i in range(3,len(self.basis)):
            B_tt = B_tt ** tntt.TT(self.basis[i](pts).transpose(), shape = [(pts.size, self.basis[i].N)])
            C_tt = C_tt ** tntt.TT(self.basis[i](pts).transpose(), shape = [(pts.size, self.basis[i].N)])
        # X = C_tt @ X
        # Y = C_tt @ Y
        # Z = C_tt @ Z
        Og_tt = C_tt @ Og_tt
        d_eval = B_tt @ self.dofs
        
        if len(self.basis)==3:
            f_eval = tntt.interpolate.function_interpolate(function, [X, Y, Z], 1e-13)
        else:
            Np = len(self.N[3:])
            meshgrid = tntt.meshgrid([tn.tensor(pts)]*Np)
            meshgrid = [X,Y,Z] + [tntt.ones([pts.size]*3)**m for m in meshgrid]
            f_eval = tntt.interpolate.function_interpolate(function, meshgrid, 1e-13)
        
            
        diff = f_eval-d_eval
        Ws = tntt.rank1TT(len(self.basis)*[tn.tensor(ws)])
        
        integral = np.abs(tntt.dot(Ws*diff,diff*Og_tt).numpy())
        # print(integral)
        return np.sqrt(integral)
    
class Geometry():
    
    def __init__(self, basis, Xs = None):
        """
        

        Args:
            basis ([type]): [description]
            Xs ([type], optional): [description]. Defaults to None.
        """
        self.N = [b.N for b in basis]
        self.basis = basis
        self.Xs = Xs
        
    def interpolate(self, geometry_map, eps = 1e-13):
        """
        Interpolates the given geometry map.

        Args:
            geometry_map (list[function]): [description]
            eps ([type], optional): [description]. Defaults to 1e-13.
        """
        Xg = [tn.tensor(b.interpolating_points()[0], dtype = tn.float64) for b in self.basis]
        Mg = [tn.tensor(b.interpolating_points()[1], dtype = tn.float64) for b in self.basis]
        
        corz = [tn.reshape(tn.linalg.inv(Mg[i]).t(), [1,Mg[i].shape[0],-1,1]) for i in range(len(Mg))]
        Gmi = tntt.TT(corz)
        
        Xs = []
        
        for i in range(3):
            evals = tntt.interpolate.function_interpolate(geometry_map[i], tntt.meshgrid(Xg), eps = eps).round(eps)
            dofs = (Gmi @ evals).round(eps)
            Xs.append(dofs)
            
        self.Xs = Xs
        
    def __call__(self, x, deriv = None):
        
        if deriv == None:
            deriv = [False] * len(self.N)
            
        Bs = [tn.tensor(self.basis[i](x[i].numpy(),derivative=deriv[i]), dtype = tn.float64).t() for i in range(len(self.N))]

        B_tt = tntt.TT([tn.reshape(m,[1,m.shape[0],-1,1]) for m in Bs])
        
        ret = []
        
        for X in self.Xs:
            ret.append(B_tt @ X)
        
        return ret

  
    
    def integral_tensor(self,eps = 1e-12):
        """
        
        
        """ 
        p1, w1 = points_basis(self.basis[0])
        p2, w2 = points_basis(self.basis[1])
        p3, w3 = points_basis(self.basis[2])
    
        cores = self.eval_omega([tn.tensor(p1),tn.tensor(p2),tn.tensor(p3)], eps).cores
        
        cores[0] = tn.einsum('ijk,j,lj->ilk',cores[0],tn.tensor(w1),tn.tensor(self.basis[0](p1)))
        cores[1] = tn.einsum('ijk,j,lj->ilk',cores[1],tn.tensor(w2),tn.tensor(self.basis[1](p2)))
        cores[2] = tn.einsum('ijk,j,lj->ilk',cores[2],tn.tensor(w3),tn.tensor(self.basis[2](p3)))
        
        return tntt.TT(cores)


class GeometryPatch():
    def __init__(self):
        pass

    def eval_omega(self, y, eps = 1e-12):
        if self.d==3 and self.dembedding==3:
            #G11, G12, G13 = self.__call__(y, 0)
            #G21, G22, G23 = self.__call__(y, 1)
            #G31, G32, G33 = self.__call__(y, 2)

            G11, G21, G31 = self.__call__(y, 0, eps = eps)
            G12, G22, G32 = self.__call__(y, 1, eps = eps)
            G13, G23, G33 = self.__call__(y, 2, eps = eps)

            det1 = G11*G22*G33
            det2 = G12*G23*G31
            det3 = G13*G21*G32
            det4 = G13*G22*G31
            det5 = G11*G32*G23
            det6 = G12*G21*G33
            res = (det1 + det2 + det3 - det4 - det5 - det6).round(eps)
        elif self.d==2:
            G11, G21 = self.__call__(y, 0, eps = eps)
            G12, G22 = self.__call__(y, 1, eps = eps)
       
            res = (G11*G22-G21*G12).round(eps)
            
        return res 

    def integral_tensor(self, basis_space, eps=1e-12):

        p1, w1 = points_basis(self.basis[0])
        p2, w2 = points_basis(self.basis[1])
        p3, w3 = points_basis(self.basis[2])
    
        cores = self.eval_omega([tn.tensor(p1),tn.tensor(p2),tn.tensor(p3)], eps).cores
        
        cores[0] = tn.einsum('ijk,j,lj->ilk',cores[0],tn.tensor(w1),tn.tensor(self.basis[0](p1)))
        cores[1] = tn.einsum('ijk,j,lj->ilk',cores[1],tn.tensor(w2),tn.tensor(self.basis[1](p2)))
        cores[2] = tn.einsum('ijk,j,lj->ilk',cores[2],tn.tensor(w3),tn.tensor(self.basis[2](p3)))
        
        return tntt.TT(cores)

    def mass_interp(self, basis_space, eps = 1e-12):
        """
        

        Args:
            eps ([type], optional): [description]. Defaults to 1e-12.

        Returns:
            [type]: [description]
        """
        ps = []
        ws = []
        
        for bs, bg in zip(basis_space, self.basis):
            if not all(x in bs.knots for x in bg.knots) or bs.deg!=bg.deg:
                pass
                # raise Exception("Knots of the FunctionSpace for the solution need to be a superset of the ones for the geometry description.")
            p, w = points_basis(bs)
            ps.append(p)
            ws.append(w)    
        
        cores = self.eval_omega(ps, eps).cores
                
        for i in range(len(self.basis)):
            if i<self.d:
                cores[i] = tn.einsum('rjs,j,mj,nj->rmns',cores[i],tn.tensor(ws[i]),tn.tensor(basis_space[i](ps[i])),tn.tensor(basis_space[i](ps[i])))
            else:
                cores[i] = tn.einsum('rjs,mj,nj->rmns',cores[i],tn.eye(cores[i].shape[1],dtype = tn.float64),tn.eye(cores[i].shape[1], dtype = tn.float64))

        return tntt.TT(cores)

    def rhs_interp(self, basis_space, function, eps = 1e-12, reference = True):

        ps = []
        ws = []
        
        for bs, bg in zip(basis_space, self.basis):
            if not all(x in bs.knots for x in bg.knots) or bs.deg!=bg.deg:
                pass
                # raise Exception("Knots of the FunctionSpace for the solution need to be a superset of the ones for the geometry description.")
            p, w = points_basis(bs)
            ps.append(tn.tensor(p,dtype=tn.float64))
            ws.append(tn.tensor(w,dtype=tn.float64))

        omega = self.eval_omega(ps, eps)   

        Xg = [tn.tensor(b.interpolating_points()[0], dtype = tn.float64) for b in self.basis]
      
        if reference:
            Xs = tntt.meshgrid(ps+[tn.ones(n) for n in self.ells])[:self.d]
        else:
            Xs = self(ps)

        if self.np==0:
            evals = tntt.interpolate.function_interpolate(function, Xs, eps)
        else:
            Np = len(self.basis[self.d:])
            meshgrid = tntt.meshgrid([x for x in Xg[self.d:]])
            meshgrid = Xs + [tntt.ones(Xs[0].N[:self.d])**m for m in meshgrid]
            # print(meshgrid)
            evals = tntt.interpolate.function_interpolate(function, meshgrid, eps, verbose = False)
            

        Bs = [tn.tensor(basis_space[i](ps[i])) for i in range(self.d)]
        Btt = tntt.rank1TT(Bs)
        Wtt = tntt.rank1TT(ws)
        if self.np!=0:
            Btt = Btt**tntt.eye(self.ells)
            Wtt = Wtt**tntt.ones(self.ells)

        return (Btt@(evals*omega*Wtt)).round(eps)

    
    def gradient_physical(self, basis_space, function, eps = 1e-12):

        ps = [tn.tensor(b.interpolating_points()[0], dtype = tn.float64) for b in basis_space]
        Ms = [tn.linalg.inv(tn.tensor(b.interpolating_points()[1], dtype = tn.float64)).T for b in basis_space]

        if self.d == 3:
            g11, g21, g31 = self.__call__(ps, 0)
            g12, g22, g32 = self.__call__(ps, 1)
            g13, g23, g33 = self.__call__(ps, 2)

        elif self.d==2:
            g11, g21 = self.__call__(ps, 0)
            g12, g22 = self.__call__(ps, 1)
            
        Og_tt = self.eval_omega(ps, eps)

        if self.d==3:
            # adjugate
            tme = datetime.datetime.now()
            h11,h12,h13 = (g22*g33-g23*g32, g13*g32-g12*g33, g12*g23-g13*g22)
            h21,h22,h23 = (g23*g31-g21*g33, g11*g33-g13*g31, g13*g21-g11*g23)
            h31,h32,h33 = (g21*g32-g22*g31, g12*g31-g11*g32, g11*g22-g12*g21)
            
            H = [[h11.round(eps),h12.round(eps),h13.round(eps)],[h21.round(eps),h22.round(eps),h23.round(eps)],[h31.round(eps),h32.round(eps),h33.round(eps)]]

            grady1 = function(ps+[tn.tensor(b.interpolating_points()[0]) for b in self.basis[self.d:]], [True,False,False]+(len(function.N)-3)*[False])
            grady2 = function(ps+[tn.tensor(b.interpolating_points()[0]) for b in self.basis[self.d:]], [False,True,False]+(len(function.N)-3)*[False])
            grady3 = function(ps+[tn.tensor(b.interpolating_points()[0]) for b in self.basis[self.d:]], [False,False,True]+(len(function.N)-3)*[False])

            grad1 = (h11*grady1+h21*grady2+h31*grady3).round(eps).mprod(Ms, [0,1,2])
            grad2 = (h12*grady1+h22*grady2+h32*grady3).round(eps).mprod(Ms, [0,1,2])
            grad3 = (h13*grady1+h23*grady2+h33*grady3).round(eps).mprod(Ms, [0,1,2])

            g1 = Function(function.basis)
            g2 = Function(function.basis)
            g3 = Function(function.basis)
            g1.dofs = grad1
            g2.dofs = grad2
            g3.dofs = grad3
            return grad1, grad2, grad3
        elif self.d==2:
            # Ogi_tt = 1/Og_tt
            # Ogi_tt = tntt.elementwise_divide(tntt.ones(Og_tt.N, dtype = tn.float64), Og_tt, eps = 1e-11, starting_tensor = None, nswp = 50, kick = 8,  verbose = False, preconditioner = 'c')
            h11, h12 = (g22,-g12)
            h21, h22 = (-g21,g11)
            h11,h12,h21,h22 = h11.round(eps),h12.round(eps),h21.round(eps),h22.round(eps)

            grady1 = function(ps+[tn.tensor(b.interpolating_points()[0]) for b in self.basis[self.d:]], [True,False]+(len(function.N)-2)*[False]).round(eps)
            grady2 = function(ps+[tn.tensor(b.interpolating_points()[0]) for b in self.basis[self.d:]], [False,True]+(len(function.N)-2)*[False]).round(eps)


            grad1 = (h11*grady1+h21*grady2).round(eps).mprod(Ms,[0,1]) / Og_tt
            grad2 = (h12*grady1+h22*grady2).round(eps).mprod(Ms,[0,1]) / Og_tt

            g1 = Function(function.basis)
            g2 = Function(function.basis)

            g1.dofs = grad1
            g2.dofs = grad2
            return g1, g2
        

        


    def stiffness_interp(self, basis_space, eps = 1e-10, func = None, func_reference = None, rankinv = 1024, device = None, verb = False, qtt = False):
        """
        

        Args:
            eps ([type], optional): [description]. Defaults to 1e-10.
            func ([type], optional): [description]. Defaults to None.
            func_reference ([type], optional): [description]. Defaults to None.
            rankinv (int, optional): [description]. Defaults to 1024.
            device ([type], optional): [description]. Defaults to None.
            verb (bool, optional): [description]. Defaults to False.
            qtt (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        
        ps = []
        ws = []
        
        for bs, bg in zip(basis_space, self.basis):
            if not all(x in bs.knots for x in bg.knots) or bs.deg!=bg.deg:
                # raise Exception("Knots of the FunctionSpace for the solution need to be a superset of the ones for the geometry description.")
                pass
            p, w = points_basis(bs, mult = 2)
            ps.append(tn.tensor(p))
            ws.append(tn.tensor(w))
        params = [tn.tensor(b.interpolating_points()[0]) for b in self.basis[self.d:]]
        
        tme = datetime.datetime.now()
        Og_tt = self.eval_omega(ps, eps)
        tme = datetime.datetime.now() - tme
        if verb: print('time omega' , tme, flush=True)
        if verb: print('rank omega,',Og_tt.R, flush = True)
        if qtt: 
            Nqtt = qtt_shape(Og_tt, list(range(len(Og_tt.N))))
            if verb:
                print('QTT enabled:')
                print(list(Og_tt.N))
                print('  || ')
                print('  \/  ')
                print(Nqtt)
            No =list(Og_tt.N)
            Og_tt = tntt.reshape(Og_tt,Nqtt)     


        if not qtt:
            tme = datetime.datetime.now()
            # Ogi_tt = 1/Og_tt
            Ogi_tt = tntt.elementwise_divide(tntt.ones(Og_tt.N, dtype = tn.float64, device = device), Og_tt, eps = eps, starting_tensor = None, nswp = 50, kick = 8,  verbose = False, preconditioner = 'c')
            tme = datetime.datetime.now() -tme
            if verb: print('time omega inv' , tme,' rank ',Ogi_tt.R,flush=True)
            #if verb: print('invert error ',(Ogi_tt*Og_tt-tt.ones(Og_tt.n)).norm()/tt.ones(Og_tt.n).norm())
            Ogi_tt = Ogi_tt.round(eps)
            #Ogi_tt = Ogi_tt.to(device)
        else:
            pass
            # Ogi_tt = tntt.elementwise_divide(tntt.ones(Og_tt.N, dtype = tn.float64, device = device), Og_tt, eps = eps, starting_tensor = None, nswp = 50, kick = 8)
        

        if func != None or func_reference != None:

            tmp = tntt.meshgrid(params) if params!=[] else []
            if func_reference == None:
                Xs = self.__call__(ps)
                F_tt = tntt.interpolate.function_interpolate(func, list(Xs)+[tntt.ones(Xs[0].N[:self.d]) ** t for t in tmp], eps = eps , verbose=False).round(eps)
                
                if verb: print('rank of Frtt is ',F_tt.r)
            else:
                F_tt = tntt.interpolate.function_interpolate(func_reference,tntt.meshgrid(ps+params),eps = eps,verbose = True).round(eps)
                if verb: print('rank of Ftt is ',F_tt.R)
        else:
            F_tt = tntt.ones(Og_tt.N)
        # F_tt = F_tt.to(device)

        if qtt:
            F_tt = tntt.reshape(F_tt,Nqtt) 
        else: 
            Ogi_tt = (Ogi_tt * F_tt).round(eps)
        
        
        if self.d == 3:
            g11, g21, g31 = self.__call__(ps, 0)
            g12, g22, g32 = self.__call__(ps, 1)
            g13, g23, g33 = self.__call__(ps, 2)
            # if device!=None:
            #     g11 = g11.to(device)
            #     g12 = g12.to(device)
            #     g13 = g13.to(device)
            #     g21 = g21.to(device)
            #     g22 = g22.to(device)
            #     g23 = g23.to(device)
            #     g31 = g31.to(device)
            #     g32 = g32.to(device)
            #     g33 = g33.to(device)
        elif self.d==2:
            g11, g21 = self.__call__(ps, 0)
            g12, g22 = self.__call__(ps, 1)
            
            # if device!=None:
            #     g11 = g11.to(device)
            #     g12 = g12.to(device)
            #     g21 = g21.to(device)
            #     g22 = g22.to(device)
    
        if self.d==3:
            # adjugate
            tme = datetime.datetime.now()
            h11,h12,h13 = (g22*g33-g23*g32, g13*g32-g12*g33, g12*g23-g13*g22)
            h21,h22,h23 = (g23*g31-g21*g33, g11*g33-g13*g31, g13*g21-g11*g23)
            h31,h32,h33 = (g21*g32-g22*g31, g12*g31-g11*g32, g11*g22-g12*g21)
            
            # if verb:
                # print(g11.R) 
                # print(g12.R) 
                # print(g13.R) 
                # print(g21.R) 
                # print(g22.R) 
                # print(g23.R) 
                # print(g31.R) 
                # print(g32.R) 
                # print(g33.R)
            # tme = datetime.datetime.now()
            H = [[h11.round(eps),h12.round(eps),h13.round(eps)],[h21.round(eps),h22.round(eps),h23.round(eps)],[h31.round(eps),h32.round(eps),h33.round(eps)]]

            tme = datetime.datetime.now() -tme
            if verb: print('H computed in' , tme)
        elif self.d==2:
            h11, h12 = (g22,-g12)
            h21, h22 = (-g21,g11)
            H = [[h11.round(eps),h12.round(eps)],[h21.round(eps),h22.round(eps)]]
            
        
        Bs = [tn.tensor(basis_space[i](ps[i]).transpose()) for i in range(self.d)]
        dBs = [tn.tensor(basis_space[i](ps[i],derivative = True).transpose()) for i in range(self.d)]
                
        N = [b.N for b in basis_space]+self.ells
        S = None
        SS = None
        Hs = dict()
        
        # the size of the bands
        band_size = [b.deg for b in basis_space[:self.d]]+[1]*self.np

        if qtt: 
            for i in range(self.d):
                for j in range(self.d):
                    H[i][j] = tntt.reshape(H[i][j], Nqtt, eps)
                    # print(H[i][j].r)

        for alpha in range(self.d):
            for beta in range(self.d):
                if verb: print('alpha, beta = ',alpha,beta)
                tme = datetime.datetime.now()

                if self.d==3:
                    tmp = H[alpha][0]*H[beta][0]+H[alpha][1]*H[beta][1]+H[alpha][2]*H[beta][2]
                elif self.d==2:
                    tmp = H[alpha][0]*H[beta][0]+H[alpha][1]*H[beta][1]
                    
                tmp = tmp.round(eps,rankinv)
                tme = datetime.datetime.now() -tme
                if verb: print('\ttime 1 ' , tme)


                tme = datetime.datetime.now()
                if not qtt:
                    tmp = tmp*Ogi_tt
                    tmp = tmp.round(eps,rankinv)
                else:
                    if device!=None and device!='cpu':
                        tmp = (tntt.elementwise_divide(tmp.to(device),Og_tt.to(device), starting_tensor = tmp.to(device), eps=eps, kick=8, nswp = 50, local_iterations = 20, resets = 4, preconditioner = 'c')*F_tt.to(device)).cpu()
                    else:
                        tmp = tntt.elementwise_divide(tmp, Og_tt, starting_tensor = tmp, eps=eps, kick=8, nswp = 50, local_iterations = 20, resets = 4, preconditioner = 'c')*F_tt
                    # tmp = tmp*Ogi_tt*F_tt

                #  print('Rank of product',tmp.r)
                
                tme = datetime.datetime.now() -tme
                if verb: print('\ttime 2 ' , tme,' rank ',tmp.R)
                
                # print('ERR ',(tmp-tmp2).norm()/tmp.norm())

                if qtt: tmp = tntt.reshape(tmp,No)
                
                # tmp = H[alpha][0]*Hi[beta][0]+H[alpha][1]*Hi[beta][1]+H[alpha][2]*Hi[beta][2]
                # tmp = tmp.round(eps,rankinv)
                
                tme = datetime.datetime.now()
                cores = tmp.cores
                
                tme = datetime.datetime.now()
                # cores[0] = np.einsum('rjs,j,jm,jn->rmns',cores[0],w1,dB1 if alpha==0 else B1,dB1 if beta==0 else B1)
                # print(cores[0].shape,w1.shape)
                for i in range(self.d):
                    cores[i] = tn.einsum('rjs,j->rjs',cores[i],ws[i])
                    tmp = tn.einsum('jm,jn->jmn',dBs[i] if alpha==i else Bs[i], dBs[i] if beta==i else Bs[i])
                    cores[i] = tn.einsum('rjs,jmn->rmns',cores[i],tmp)
                    
                for i in range(self.d,len(cores)):
                    cores[i] = tn.einsum('rjs,mj,nj->rmns',cores[i],tn.eye(cores[i].shape[1],dtype=tn.float64),tn.eye(cores[i].shape[1],dtype=tn.float64))
                tme = datetime.datetime.now() -tme
                if verb: print('\t\ttime ' , tme)
                
                
                tme = datetime.datetime.now()
                
                ss = tntt.TT([tn.tensor(bandcore2ttcore(cores[i].cpu().numpy(),band_size[i])) for i in range(len(cores))]).to(device)

                
                SS = ss if SS==None else SS+ss 
                
                
                tme = datetime.datetime.now() -tme
                if verb: print('\ttime 4 ' , tme)
            tme = datetime.datetime.now()

            SS = SS.round(eps)
            
            tme = datetime.datetime.now() -tme
            if verb: print('\ttime ROUND ' , tme)
        
        cores = SS.cores
        SS = tntt.TT([tn.tensor(ttcore2bandcore(cores[i].cpu().numpy(),N[i],band_size[i])) for i in range(len(cores))])

        return SS
        
    def plot_domain(self, params = None, bounds = None, fig = None, wireframe = True, frame_color = 'r', n = 12, surface_color = 'blue',alpha = 0.4, line_width = 1.0):
        """
        Plot the domain for a given parameter (if any parameter dependence exists).
        Args:
            params ([type], optional): [description]. Defaults to None.
            bounds ([type], optional): [description]. Defaults to None.
            fig ([type], optional): [description]. Defaults to None.
            wireframe (bool, optional): [description]. Defaults to True.
            frame_color (str, optional): [description]. Defaults to 'r'.
            n (int, optional): [description]. Defaults to 12.
            surface_color (str, optional): [description]. Defaults to 'blue'.
            alpha (float, optional): [description]. Defaults to 0.4.
        Returns:
            [type]: [description]
        """
        if fig == None:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
        else:
            ax = fig.gca()
            
        if wireframe:
            plot_func = ax.plot_wireframe
        else:
            plot_func = ax.plot_surface
        
        if bounds == None:
            bounds = [b.interval for b in self.basis[:3]]
            
        if surface_color != None:
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][1],n, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][1],n, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][0],1, dtype = tn.float64)]+([] if params ==None else params))
            plot_func(x.full()[:,:,0].numpy().squeeze(), y.full()[:,:,0].numpy().squeeze(), z.full()[:,:,0].numpy().squeeze(), color = surface_color,alpha = alpha)
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][1],n, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][1],n, dtype = tn.float64),tn.linspace(bounds[2][1],bounds[2][1],1, dtype = tn.float64)]+([] if params ==None else params))
            plot_func(x.full()[:,:,0].numpy().squeeze(), y.full()[:,:,0].numpy().squeeze(), z.full()[:,:,0].numpy().squeeze(), color = surface_color,alpha = alpha)
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][1],n, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][0],1, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][1],n, dtype = tn.float64)]+([] if params ==None else params))
            plot_func(x.full()[:,0,:].numpy().squeeze(), y.full()[:,0,:].numpy().squeeze(), z.full()[:,0,:].numpy().squeeze(), color = surface_color,alpha = alpha)
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][1],n, dtype = tn.float64),tn.linspace(bounds[1][1],bounds[1][1],1, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][1],n, dtype = tn.float64)]+([] if params ==None else params))
            plot_func(x.full()[:,0,:].numpy().squeeze(), y.full()[:,0,:].numpy().squeeze(), z.full()[:,0,:].numpy().squeeze(), color = surface_color,alpha = alpha)
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][0],1, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][1],n, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][1],n, dtype = tn.float64)]+([] if params ==None else params))
            plot_func(x.full()[0,:,:].numpy().squeeze(), y.full()[0,:,:].numpy().squeeze(), z.full()[0,:,:].numpy().squeeze(), color = surface_color,alpha = alpha)
            
            x,y,z = self([tn.linspace(bounds[0][1],bounds[0][1],1, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][1],n, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][1],n, dtype = tn.float64)]+([] if params ==None else params))
            plot_func(x.full()[0,:,:].numpy().squeeze(), y.full()[0,:,:].numpy().squeeze(), z.full()[0,:,:].numpy().squeeze(), color = surface_color,alpha = alpha)
        
        if frame_color != None:
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][1],n, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][0],1, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][0],1, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color, linewidth = line_width)
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][1],n, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][0],1, dtype = tn.float64),tn.linspace(bounds[2][1],bounds[2][1],1, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color, linewidth = line_width)
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][1],n, dtype = tn.float64),tn.linspace(bounds[1][1],bounds[1][1],1, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][0],1, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color, linewidth = line_width)
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][1],n, dtype = tn.float64),tn.linspace(bounds[1][1],bounds[1][1],1, dtype = tn.float64),tn.linspace(bounds[2][1],bounds[2][1],1, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color, linewidth = line_width)
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][0],1, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][1],n, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][0],1, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color, linewidth = line_width)
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][0],1, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][1],n, dtype = tn.float64),tn.linspace(bounds[2][1],bounds[2][1],1, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color, linewidth = line_width)
            
            x,y,z = self([tn.linspace(bounds[0][1],bounds[0][1],1, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][1],n, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][0],1, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color, linewidth = line_width)
            
            x,y,z = self([tn.linspace(bounds[0][1],bounds[0][1],1, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][1],n, dtype = tn.float64),tn.linspace(bounds[2][1],bounds[2][1],1, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color, linewidth = line_width)
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][0],1, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][0],1, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][1],n, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color, linewidth = line_width)
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][0],1, dtype = tn.float64),tn.linspace(bounds[1][1],bounds[1][1],1, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][1],n, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color, linewidth = line_width)
            
            x,y,z = self([tn.linspace(bounds[0][1],bounds[0][1],1, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][0],1, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][1],n, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color, linewidth = line_width)
            
            x,y,z = self([tn.linspace(bounds[0][1],bounds[0][1],1, dtype = tn.float64),tn.linspace(bounds[1][1],bounds[1][1],1, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][1],n, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color, linewidth = line_width)
        
        return fig

class PatchNURBS(GeometryPatch):

    def __init__(self, basis, basis_param, control_points, weights, bounds = None):
        # super(PatchNURBS, self).__init__()
        self.d = len(basis)
        self.dembedding = len(control_points)
        self.np = len(basis_param)
        self.basis = basis.copy()+basis_param.copy()
        self.control_points = control_points.copy()
        self.ells = [b.N for b in basis_param]
        
        if len(weights.N) == self.d+self.np and self.np>0:
            self.weights = weights
        else: 
            self.weights = weights**tntt.ones(self.ells)
            
        if bounds == None:
            self.bounds = [(b.interval[0],b.interval[-1]) for b in self.basis]
        else:
            self.bounds = bounds
            
    def __call__(self, y, derivative = None, eps = 1e-14):
        if len(y) == self.d:
            Bs = [tn.tensor(self.basis[i](y[i])).t() for i in range(self.d)]
            Btt = tntt.TT([mat[None,...,None] for mat in Bs]) ** (None if self.np==0 else tntt.eye(self.ells))
        if len(y) == self.d+self.np:
            Bs = [tn.tensor(self.basis[i](y[i])).t() for i in range(self.d+self.np)]
            Btt = tntt.rank1TT(Bs)
        den = Btt @ self.weights # self.weights.mprod(Bs, list(range(self.d)))
        
        if all([e==1 for e in den.R]):
            deninv = tntt.TT([1/c for c in den.cores])
        else:
            deninv = (1/den)
            

        result = []

        if derivative == None:

            for i, ctl in enumerate(self.control_points):
                
                tmp = Btt @ tntt.diag(self.weights)
                result.append( (tmp @ ctl) * deninv )
        else:
            dBs = [tn.tensor(self.basis[i](y[i], derivative = derivative==i)).t() for i in range(self.d)]
            dBtt = tntt.rank1TT(dBs) ** (None if self.np==0 else tntt.eye(self.ells))

            for ctl in self.control_points:
                tmp = (dBtt @ (self.weights*ctl))*den- (Btt @ (self.weights*ctl)) * (dBtt @ self.weights)
                result.append(tmp.round(eps)*deninv*deninv)
      
        return [r.round(eps) for r in result]
                
    def __repr__(self):
        if self.d == 1:
            s = 'NURBS curve'
        elif self.d == 2:
            s = 'NURBS surface'
        elif self.d == 3:
            s = 'NURBS volume'
        else: 
            s = 'NURBS instance'
        
        s += ' embedded in a '+str(self.dembedding)+'D space.\n'
        s += 'Basis:\n' 
        for b in self.basis:
            s+=str(b)+'\n'
        
        return s
        
    def __getitem__(self, key):
         
        if len(key) != self.d+self.np:
            raise Exception('Invalid number of dimensions.')
        
        basis_new = []
        basis_new_param = []
        bounds_new = []
        
        weights = self.weights.clone()
        
        axes = [] 
        for k, id in enumerate(key):

            if isinstance(id,int) or isinstance(id,float):
                if self.bounds[k][0]<=id and id<=self.bounds[k][1]:
                    axes.append(k)
                    
                    B = self.basis[k](id).flatten()
                    s = tuple([None]*k+[slice(None,None,None)]+[None]*(self.d-k-1))
                    B = B[s]
                    weights = weights*B
                    
                else:
                    raise Exception("Value must be inside the domain of the BSpline basis.")
            elif isinstance(id,slice):
                basis_new.append(self.basis[k])
                start = id.start if id.start!=None else self.bounds[k][0]
                stop = id.stop if id.stop!=None else self.bounds[k][1]
                bounds_new.append((start,stop))
            else:
                raise Exception("Only scalars are permitted")

        weights_new = np.sum(weights,axis=tuple(axes))
        knots_new = np.sum(self.control_points*weights[...,None], axis=tuple(axes))
        knots_new = knots_new/weights_new[...,None]
        
               
        return PatchNURBS(basis_new,knots_new, weights_new, self.rand_key, bounds=bounds_new)
    
    @staticmethod
    def interpolate_parameter_dependent(control_points_function, weights_function, basis_solution, basis_param, eps = 1e-12):
        
        parameter_grid = [b.interpolating_points()[0] for b in basis_param]
        Ns = [b.N for b in basis_solution]
        Nps = [b.N for b in basis_param]
        d = len(Ns)        

        
        
        faux = lambda I: weights_function[tuple(I[:(d)])](tn.tensor( [parameter_grid[k][idx] for k, idx in enumerate(I[d:])] )) if callable(weights_function[tuple(I[:(d)])]) else weights_function[tuple(I[:(d)])]
        faux2 =lambda I: tn.tensor([faux(i) for i in I])
        weights = tntt.interpolate.dmrg_cross(faux2, Ns+Nps, eps, eval_vect = True).round(eps)

        faux = lambda I: control_points_function[tuple(I[:(d+1)])](tn.tensor( [parameter_grid[k][idx] for k, idx in enumerate(I[(d+1):])] )) if callable(control_points_function[tuple(I[:(d+1)])]) else control_points_function[tuple(I[:(d+1)])]
        faux2 =lambda I: tn.tensor([float(faux(i)) for i in I])
        control_points = tntt.interpolate.dmrg_cross(faux2, [d]+Ns+Nps, eps, eval_vect = True).round(eps)
        
        control_points = [control_points[k,...].round(eps) for k in range(control_points.N[0])]

        return PatchNURBS(basis_solution, basis_param, control_points, weights)
    
class PatchBSpline(GeometryPatch):

    def __init__(self, basis, basis_param, control_points = None):
        # super(PatchNURBS, self).__init__()
        self.d = len(basis)
        self.dembedding = len(control_points)
        self.np = len(basis_param)
        self.basis = basis.copy()+basis_param.copy()
        self.control_points = control_points.copy()
        self.ells = [b.N for b in basis_param]
        
        
            
    def __call__(self, y, derivative = None, eps = 1e-15):
        
            
        Bs = [tn.tensor(self.basis[i](y[i], derivative = derivative==i)).t() for i in range(len(y))]
        if self.d == len(y):
            Btt = tntt.rank1TT(Bs) ** (None if self.np==0 else tntt.eye(self.control_points[0].N[self.d:]))
        else:
            Btt = tntt.rank1TT(Bs)
            
        result = []
        
        for X in self.control_points:
            result.append(Btt @ X)
        
        
        return result
                
    @staticmethod
    def interpolate_geometry(geometry_map, basis_geometry, basis_params, eps = 1e-13):
        Xg = [tn.tensor(b.interpolating_points()[0], dtype = tn.float64) for b in basis_geometry] + [tn.tensor(b.interpolating_points()[0], dtype = tn.float64) for b in basis_params]
        Mg = [tn.tensor(b.interpolating_points()[1], dtype = tn.float64) for b in basis_geometry] + [tn.tensor(b.interpolating_points()[1], dtype = tn.float64) for b in basis_params]
        
        corz = [tn.reshape(tn.linalg.inv(Mg[i]).t(), [1,Mg[i].shape[0],-1,1]) for i in range(len(Mg))]
        Gmi = tntt.TT(corz)
        
        Xs = []
        
        for i in range(len(geometry_map)):
            evals = tntt.interpolate.function_interpolate(geometry_map[i], tntt.meshgrid(Xg), eps = eps).round(eps)
            dofs = (Gmi @ evals).round(eps)
            Xs.append(dofs)
            
        return PatchBSpline(basis_geometry, basis_params, Xs)

    def __repr__(self):
        if self.d == 1:
            s = 'BSpline curve'
        elif self.d == 2:
            s = 'Bspline surface'
        elif self.d == 3:
            s = 'Bspline volume'
        else: 
            s = 'BSpline instance'
        
        s += ' embedded in a '+str(self.dembedding)+'D space.\n'
        s += 'Basis:\n' 
        for b in self.basis:
            s+=str(b)+'\n'
        
        return s
        
    def __getitem__(self, key):
         
        if len(key) != self.d:
            raise Exception('Invalid number of dimensions.')
        
        basis_new = []
        basis_new_param = []
        control_points = [x.clone() for x in self.control_points]
        
        axes = [] 
        for k, id in enumerate(key):
            if isinstance(id,int) or isinstance(id,float):
                if self.basis[k].interval[0]<=id and id<=self.basis[k].interval[1]:
                    axes.append(k)
                    Bmat = self.basis[k](np.array([float(id)]))
                    for i in range(len(control_points)):
                        control_points[i].mprod(Bmat.T)
                else:
                    raise Exception("Value must be inside the domain of the BSpline basis.")
            elif isinstance(id,slice) and id.start == None and id.stop==None and id.stop == None:
                if k < self.d:
                    basis_new.append(self.basis[k])
                else:
                    basis_new_param.append(self.basis[k])
            else:
                raise Exception("Only slices and scalars are permitted")

        for i in range(len(self.control_points)):
            control_points[i] = control_points[i].sum(axis=tuple(axes))
        
        
               
        return PatchBSpline(basis_new, basis_new_param, control_points)


# 
    
