import os, time, random, abc
import numpy as np
import tensorflow as tf

from collections import deque
from models.base import BaseModel


##TODO 
class Agent(BaseModel):
    __metaclass__ = abc.ABCMeta

    ''' Agent Abstract Class ''' 

    def __init__(self, config):
        super(Agent, self).__init__(config)

        pass


    @abc.abstractmethod
    def _build_graph(self):
        """Build appropriate agent network"""
        pass


    @abc.abstractmethod
    def get_action(self, state):
        
        pass



