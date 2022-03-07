from abc import ABC, abstractmethod

class MethodTemplate(ABC):

    @abstractmethod
    def __init__(self):
        ...

    @abstractmethod
    def method(self):
        ...

    @property
    def id(self):
        return self._id
    
    @id.setter
    def id(self, id):
        self._id = id
    
    @property
    def task(self):
        return self._task
    
    @task.setter
    def task(self, task):
        self._task = task

    def get_parameters(self):
        return [None,]



        