import fenics as fn
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import pickle

def create_file_and_mesh(theta,meshsize = 0.5):

    with open('./examples/fem_mesh/wg_proto.geo', 'r') as file:
        data = file.read()

    s = "theta1 = %.18f; \ntheta1 = %.18f; \ntheta3 = %.18f;\ntheta3 = %.18f;\nmeshsize=%.18f;"%(theta[0],theta[1],theta[2],theta[3],meshsize)

    s = s + data

    with  open("tmp.geo", "w") as file:
        file.write(s)
        file.close()

    os.system('gmsh tmp.geo -3 -o tmp.msh -format msh2 >/dev/null 2>&1')

    os.system('dolfin-convert tmp.msh tmp.xml >/dev/null 2>&1')

    mesh = fn.Mesh('tmp.xml')
    markers = fn.MeshFunction("size_t", mesh, 'tmp_physical_region.xml')
    boundaries = fn.MeshFunction('size_t', mesh, 'tmp_facet_region.xml')

    return mesh, markers, boundaries



class SolverWG():

    def __init__(self):
        pass

    def set_params(self, theta, meshsize=0.4):
        '''
        Set the parameters.

        Parameters
        ----------
        theta : list of floats or numpy array
            The parameters. Belong to [-0.05,0.05].
        meshsize : float, optional
            The meshgrid size. The default is 0.4.

        Returns
        -------
        None.

        '''
        self.theta = theta
        self.meshsize = meshsize

    def create_mesh(self):
        '''
        Create the mesh and save it

        Returns
        -------
        tme : datetime object
            Duration of simulation.

        '''
        tme = datetime.datetime.now()
        mesh, subdomains, boundaries = create_file_and_mesh(self.theta, self.meshsize)
        self.mesh = mesh
        self.subdomains = subdomains
        self.boundaries = boundaries
        tme = datetime.datetime.now() - tme
        return tme

    def solve_laplace(self, neumann = False):
        '''
        Solve the problem

        Returns
        -------
        tme : datetime object
            Duration of simulation.

        '''
        tme = datetime.datetime.now()


        class mat(fn.UserExpression):
            def __init__(self, markers, val, **kwargs):
                self.markers = markers
                self.val = val
                super().__init__(**kwargs)
            
            def eval_cell(self, values, x, cell):
                if self.markers[cell.index] == 105:
                    values[0] = self.val
                else:
                    values[0] = 1
    
        kappa = mat(self.subdomains, self.theta[3], degree=2)


        tme1 = datetime.datetime.now()
        dx = fn.Measure('dx', domain=self.mesh, subdomain_data=self.subdomains)
        V = fn.FunctionSpace(self.mesh, 'CG', 1)


        if neumann:
            top_boundary = fn.DirichletBC(V, fn.Constant(0.0), self.boundaries, 102)
            bottom_boundary = fn.DirichletBC(V, fn.Constant(10.0), self.boundaries, 101)
            bcs =[top_boundary, bottom_boundary]
        else:
            top_boundary = fn.DirichletBC(V, fn.Constant(0.0), self.boundaries, 102)
            bottom_boundary = fn.DirichletBC(V, fn.Constant(0.0), self.boundaries, 101)
            pec_boundary = fn.DirichletBC(V, fn.Constant(0.0), self.boundaries, 104)
            bcs =[top_boundary, bottom_boundary,pec_boundary]


        # Solve the Poisson equation with the source set to 0
        u = fn.TrialFunction(V)
        v = fn.TestFunction(V)
        a = fn.dot(fn.grad(u), fn.grad(v)) * kappa * fn.dx
        if neumann:
            L = fn.Constant('0') * v * fn.dx
        else:
            L = fn.Constant('1') * v * fn.dx
        u = fn.Function(V)
        tme1 = datetime.datetime.now() - tme1
        print('Time spaces ',tme1)

        tme2 = datetime.datetime.now()
        fn.solve(a == L, u, bcs, solver_parameters={str('linear_solver'): str('gmres')})
        tme2 = datetime.datetime.now() - tme2
        print('Time solve ',tme2)
        #problem = fn.LinearVariationalProblem(a, L, u, bcs)
        #solver = fn.LinearVariationalSolver(problem)
        # fn.solve(a == L, u, bcs)

        self.u = u

        tme = datetime.datetime.now() - tme
        return tme1, tme2

    def solve_helmholz(self,k = 49):
        '''
        Solve the problem

        Returns
        -------
        tme : datetime object
            Duration of simulation.

        '''
        tme = datetime.datetime.now()



        tme1 = datetime.datetime.now()
        dx = fn.Measure('dx', domain=self.mesh, subdomain_data=self.subdomains)
        V = fn.FunctionSpace(self.mesh, 'CG', 1)

        class mat(fn.UserExpression):
            def __init__(self, markers, val, **kwargs):
                self.markers = markers
                self.val = val
                super().__init__(**kwargs)
            
            def eval_cell(self, values, x, cell):
                if self.markers[cell.index] == 105:
                    values[0] = self.val
                else:
                    values[0] = 1
    
        kappa = mat(self.subdomains, self.theta[3], degree=2)

        top_boundary = fn.DirichletBC(V, fn.Constant(0.0), self.boundaries, 102)
        bottom_boundary = fn.DirichletBC(V, fn.Expression('cos(x[0]*1.5707963267948966)*sin(x[1]*6.283185307179586)', degree=1), self.boundaries, 101)
        pec_boundary = fn.DirichletBC(V, fn.Constant(0.0), self.boundaries, 104)
        bcs =[top_boundary, bottom_boundary,pec_boundary]


        # Solve the Poisson equation with the source set to 0
        u = fn.TrialFunction(V)
        v = fn.TestFunction(V)
        a = fn.dot(fn.grad(u), fn.grad(v)) * kappa* fn.dx - fn.Constant(k)*u*v*fn.dx

        L = fn.Constant('0') * v * fn.dx

        u = fn.Function(V)
        tme1 = datetime.datetime.now() - tme1
        print('Time spaces ',tme1)

        tme3 = datetime.datetime.now()
        A = fn.assemble(a)
        #rows, cols, values = A.data()
        #Aa = sps.csr_matrix((values, cols, rows))
        b = fn.assemble(L)
        #print('\tshape ',Aa.shape)
        tme3 = datetime.datetime.now() - tme3
        print('Time assemble ',tme3)

        tme2 = datetime.datetime.now()
        fn.solve(a == L, u, bcs, solver_parameters={str('linear_solver'): str('gmres')})
        tme2 = datetime.datetime.now() - tme2
        print('Time solve ',tme2)
        #problem = fn.LinearVariationalProblem(a, L, u, bcs)
        #solver = fn.LinearVariationalSolver(problem)
        # fn.solve(a == L, u, bcs)

        self.u = u

        tme = datetime.datetime.now() - tme
        return tme1, tme2

    def get_dof_vector(self):
        '''
        Returns the DoF vector of the solution.

        Returns
        -------
        numpy array
            the DoF vector.

        '''

        return self.u.vector()

    def get_dof_size(self):
        '''
        Returns the size of the DoF vector.

        Returns
        -------
        int
            the size of the DoF vector.

        '''

        return self.u.vector()[:].size

    def __call__(self, x1s, x2s, x3s):
        '''
        Evaluates the solution.

        Parameters
        ----------
        x1s : numpy array
            first coordinates.
        x2s : numpy array
            second coordinates.
        x3s : numpy array
            third coordinates.

        Returns
        -------
        numpy array
            the solution evaluated on the given points.

        '''
        shape = x1s.shape

        x1s = x1s.flatten()
        x2s = x2s.flatten()
        x3s = x3s.flatten()

        ucalc = 0*x1s

        for i in range(x1s.size):
            try:
                ucalc[i] = self.u((x1s[i],x2s[i],x3s[i]))
            except:
                ucalc[i] = np.nan
        return ucalc.reshape(shape)






if __name__ == "__main__":
    # load TT-IGA


    solver_fine = SolverWG()
    solver_fine.set_params([0.0]*3+[1.0],0.1/4)
    tme_mesh = solver_fine.create_mesh()
    # tme_solve = solver_fine.solve_laplace()

    # x,y,z = np.meshgrid(np.linspace(-1.5,1.5,128),np.array([0.25]),np.linspace(-3,1,128))
    # ufem = solver_fine(x,y,z)

    # plt.figure()
    # plt.contourf(x.squeeze(),z.squeeze(),ufem.squeeze(),levels=128)
    # # plt.title('FEM')
    # plt.xlabel(r'$x_1$')
    # plt.ylabel(r'$x_3$')
    # plt.colorbar()


    tme_solve = solver_fine.solve_helmholz(64)

    x,y,z = np.meshgrid(np.linspace(-1.5,1.5,128),np.array([0.25]),np.linspace(-3,0,128))
    ufem = solver_fine(x,y,z)

    plt.figure()
    plt.contourf(x.squeeze(),z.squeeze(),ufem.squeeze(),levels=128)
    # plt.title('FEM')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_3$')
    plt.colorbar()

    print('max ',np.nanmax(ufem),' min ',np.nanmin(ufem))
    x,y,z = np.meshgrid(np.linspace(-1.5,1.5,384),np.linspace(0,0.5,64),np.array([-3]))
    ufem = solver_fine(x,y,z)

    plt.figure()
    plt.contourf(x.squeeze(),y.squeeze(),ufem.squeeze(),levels=128)
    # plt.title('FEM')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.colorbar()

    print('DoF size ',solver_fine.get_dof_size())

    plt.show()


