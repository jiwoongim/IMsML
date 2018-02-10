import os, time, random
import numpy as np
import tensorflow as tf

class Optimizer(object):
    

    def __init__(self, opt_config):

        self.lr = opt_config['lr']


    def min_func(self, model):

        gradients = self.compute_gradients(model.J)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, 5), var)

        # Add histograms for gradients.
        for grad, var in gradients:
            tf.histogram_summary(var.name, var)
            if grad is not None:
                tf.histogram_summary(var.name + '/gradients', grad)

        return self.apply_gradients(gradients)


    def min_func(self, model):

        gradients = self.compute_gradients(model.J)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, 5), var)

        # Add histograms for gradients.
        for grad, var in gradients:
            tf.histogram_summary(var.name, var)
            if grad is not None:
                tf.histogram_summary(var.name + '/gradients', grad)

        return self.apply_gradients(gradients)


