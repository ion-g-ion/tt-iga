import torch as tn
import numpy as np
import torchtt as tntt

class UnivariateBasis():

    pass

class FunctionSpaceTP():
    def __init__(self, functions = []):
        if isinstance(functions, list):
            self.__basis = [b.copy() for b in functions]
            self.__N = tuple([b.N for b in self.__basis])
            self.__d = len(self._N)
        elif isinstance(functions, UnivariateBasis):
            self.__basis = [functions.copy()]
            self.__N = tuple([self.__basis[0].N])
            self.__d = 1
        else:
            raise Exception("Invalid arguments.")
        
    @property
    def dim(self):
        """
        The dimension of the bais

        Returns:
            tuple: the dimension of the basis.
        """
        return self.__N
    
    @property
    def basis(self):
        """
        Return the bases forming the tensor product space.

        Returns:
            list[UnivaraiteBasis]: _description_
        """
        
        return self.__basis
    
    def __repr__(self):
        """
        Represent the object as a string.

        Returns:
            str: the string representation.
        """
        
        s = "Tensor-product function space with bases:\n"
        for b in self.__basis:
            s+=str(b)+'\n'
        return s
    
    def __pow__(self, other):
        """
        Implement the Kronecker product between two spaces.
        The dimension is mutipled.

        Args:
            other (Union[functionSpaceTP,UnivariateBasis]): _description_

        Raises:
            Exception: Kronecker product `**` is possible only between two tensor spaces or a tensor space and an univariate basis.

        Returns:
            FunctionSpaceTP: the resulting space.
        """
        if isinstance(other, FunctionSpaceTP):
            return FunctionSpaceTP([b.copy() for b in self.__basis ]+[b.copy() for b in other.basis])
        elif isinstance(other, UnivariateBasis):
            return FunctionSpaceTP([b.copy() for b in self.__basis ]+[other.copy()])
        else:
            raise Exception("Kronecker product `**` is possible only between two tensor spaces or a tensor space and an univariate basis.")
        
    def __rpow__(self, other):
        """
        Implement the Kronecker product between two spaces.
        The dimension is mutipled.

        Args:
            other (Union[functionSpaceTP,UnivariateBasis]): _description_

        Raises:
            Exception: Kronecker product `**` is possible only between two tensor spaces or a tensor space and an univariate basis.

        Returns:
            FunctionSpaceTP: the resulting space.
        """
        if isinstance(other, FunctionSpaceTP):
            return FunctionSpaceTP([b.copy() for b in other.basis]+[b.copy() for b in self.__basis])
        elif isinstance(other, UnivariateBasis):
            return FunctionSpaceTP([other.copy()]+[b.copy() for b in self.__basis ])
        else:
            raise Exception("Kronecker product `**` is possible only between two tensor spaces or a tensor space and an univariate basis.")
        
    def interpolate(self, func, composition_map = None):
        
        pass
    
    def __getitem__(self, index):
        
        return type(self)(self.__basis[index].copy())
    
    
    # def eval(self, dofs, points):
    #     """
    #     Evaluate a function of the function space on a meshgrid.

    #     Args:
    #         dofs (torchtt.TT): the dofs of the function from the given function space.
    #         points (list[torch.tensor]): list containing the univariate points of the meshgrid where the function given by the dofs tensor has to be evaluated.

    #     Raises:
    #         Exception: Invalid argumen: points must be a list of length equal to the number of dimensions.

    #     Returns:
    #         torchtt.TT: the evaluation.
    #     """
    #     
    #     if len(points)!=self._d:
    #         raise Exception("Invalid argumen: points must be a list of length equal to the number of dimensions.")
    #     
    #     
    #     Bs = [tn.tensor(b(p).T) for b,p in zip(self.basis,points)]
    #     return tntt.rank1TT(Bs) @ dofs

    def quadrature_points(self, mult = 2):
        return 1,1