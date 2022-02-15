from abc import ABC, abstractmethod

class MethodTemplate(ABC):

    def __init__(self, task, id):
        self.task = task
        self.id = id

    @abstractmethod
    def method(self):
        ...

    def get_method_id(self):
        return self.id
    
    def get_task(self):
        return self.task

        