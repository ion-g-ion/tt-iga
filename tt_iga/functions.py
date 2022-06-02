import torch as tn
import numpy as np
import torchtt as tntt

class UnivariateBasis():

    pass

class FunctionSpaceTP():
    def __init__(self, functions):
        if isinstance(functions, list):
            self.basis = [b.copy() for b in functions]
            self._N = [b.N for b in self.basis]
            self._d = len(self._N)
        elif isinstance(functions, UnivariateBasis):
            self.basis = [functions.copy()]
            self._N = [self.basis[0].N]
            self._d = 1
        else:
            raise Exception("Invalid arguments.")
        
    def __repr__(self):
        """
        Represent the object as a string.

        Returns:
            str: the string representation.
        """
        
        s = "Tensor-product function space with bases:\n"
        for b in self.basis:
            s+=str(b)+'\n'
        return s
    
    def eval(self, dofs, points):
        """
        Evaluate a function of the function space on a meshgrid.

        Args:
            dofs (torchtt.TT): the dofs of the function from the given function space.
            points (list[torch.tensor]): list containing the univariate points of the meshgrid where the function given by the dofs tensor has to be evaluated.

        Raises:
            Exception: Invalid argumen: points must be a list of length equal to the number of dimensions.

        Returns:
            torchtt.TT: the evaluation.
        """
        
        if len(points)!=self._d:
            raise Exception("Invalid argumen: points must be a list of length equal to the number of dimensions.")
        
        
        Bs = [tn.tensor(b(p).T) for b,p in zip(self.basis,points)]
        return tntt.rank1TT(Bs) @ dofs
    
    def interpolate(self, func):
        
        dofs = 1
        return dofs
    
    def quadrature(self, mult = 2):
        return 1,1