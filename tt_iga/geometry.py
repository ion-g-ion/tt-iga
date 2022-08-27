import numpy as np

class ParameterDependentControlPoints():

    def __init__(self,N):
        self.N = N.copy()
        self.__arr = np.zeros([len(N)]+N, dtype = object)

    def __setitem__(self, key, value):
        self.__arr[key] = value
        
    def __getitem__(self, key):

        return self.__arr[key]

    def __call__(self, x):
        arr = self.__arr.flatten()
        result = np.zeros(arr.size)
        for i in range(arr.size):
            result[i] = float(arr[i](x)) if callable(arr[i]) else arr[i]
        result = result.reshape(self.__arr.shape)
        return result

class ParameterDependentWeights():

    def __init__(self,N):
        self.N = N.copy()
        self.__arr = np.zeros(N, dtype = object)
    def __setitem__(self, key, value):

        self.__arr[key] = value
        
    def __getitem__(self, key):
        return self.__arr[key]

    def __call__(self, x):
        arr = self.__arr.flatten()
        result = np.zeros(arr.size)
        for i in range(arr.size):
            result[i] = float(arr[i](x)) if callable(arr[i]) else arr[i]
        result = result.reshape(self.__arr.shape)
        return result

