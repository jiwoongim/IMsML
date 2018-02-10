from abc import ABCMeta, abstractmethod

class Base( ):
    __metaclass__ = ABCMeta
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def run(self, data):
        """
        this runs operations on data, and returns new dataset.
        """
        return
