import tensorflow as tf



def copy_variables(variables):
    """Code from https://github.com/nivwusquorum/tensorflow-deepq/blob/master/tf_rl/utils/__init__.py"""


    res = {}
    for v in variables:
        name = base_name(v)
        copied_var = tf.Variable(v.initialized_value(), name=name)
        res[name] = copied_var
    return res


