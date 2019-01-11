'''
Instantiating new porject with specific solver
'''
from .io import backend

from inspect import isclass, isabstract, getmembers
import pymania.solvers as solvers


class Factory:
    def __init__(self):
        self.svs = {} # key: solver class name and value its class
        classes = getmembers(solvers,lambda x:isclass(x))
        for name,_type in classes:
            if isclass(_type) and issubclass(_type,solvers.Solver):
                self.svs[name] = _type
        self._backend = backend()

    @property
    def backend(self):
        return self._backend

F = Factory()
print(F.svs)

def create_project(name,id):
    if name in F.svs:
        return F.svs[name](F.backend,id)
    else:
        raise Exception('Solver does not exist!')
