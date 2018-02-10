import os, time, random, cPickle
import numpy as np
import tensorflow as tf


def unpickle(path):
    ''' For cifar-10 data, it will return dictionary'''
    #Load the cifar 10
    f = open(path, 'rb')
    data = cPickle.load(f)
    f.close()
    return data 


def pickle(x,fname):
    '''save pickled weights'''
    f = file(fname+'.pkl', 'wb')
    cPickle.dump(x, f, protocol=cPickle.HIGHEST_PROTOCOL)
    # print("saved!")
    f.close()



def checkpoint(sess):

    save_path = saver.save(sess, "./model.ckpt")
    print("Model saved in file: %s" % save_path)


def restore_model(sess):
    saver = tf.train.Saver()
    if os.path.exists("model.ckpt"):
	saver.restore(sess, "model.ckpt")
	print("Model restored.")


#def last_relevant(output, length):
#    batch_size = tf.shape(output)[1]
#    max_length = int(output.get_shape()[0])
#    output_size = int(output.get_shape()[2])
#    index = tf.range(0, batch_size) * max_length + (length - 1)
#    flat = tf.reshape(output, [-1, output_size])
#    relevant = tf.gather(flat, index)
#    return relevant


def last_relevant(output, length):
    
    ##NOTE : use extract_last_relevant
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


def extract_last_relevant(outputs, length):
    """
    Args:
        outputs: [Tensor(batch_size, output_neurons)]: A list containing the output
            activations of each in the batch for each time step as returned by
            tensorflow.models.rnn.rnn.
        length: Tensor(batch_size): The used sequence length of each example in the
            batch with all later time steps being zeros. Should be of type tf.int32.

    Returns:
        Tensor(batch_size, output_neurons): The last relevant output activation for
            each example in the batch.
    """
    output = tf.transpose(tf.pack(outputs), perm=[1, 0, 2])
    # Query shape.
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    num_neurons = int(output.get_shape()[2])
    # Index into flattened array as a workaround.
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, num_neurons])
    relevant = tf.gather(flat, index)
    return relevant

def gather_2d(params, indices):
    # only for two dim now
    shape = params.get_shape().as_list()
    assert len(shape) == 2, 'only support 2d matrix'
    flat = tf.reshape(params, [np.prod(shape)])
    flat_idx = tf.slice(indices, [0,0], [shape[0],1]) * shape[1] + tf.slice(indices, [0,1], [shape[0],1])
    flat_idx = tf.reshape(flat_idx, [flat_idx.get_shape().as_list()[0]])
    return tf.gather(flat, flat_idx)

"""Utility for displaying Tensorflow graphs from:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
"""
import tensorflow as tf
import numpy as np
from IPython.display import display, HTML


def show_graph(graph_def):
    # Helper functions for TF Graph visualization
    def _strip_consts(graph_def, max_const_size=32):
        """Strip large constant values from graph_def."""
        strip_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = strip_def.node.add()
            n.MergeFrom(n0)
            if n.op == 'Const':
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
                if size > max_const_size:
                    tensor.tensor_content = "<stripped {} bytes>".format(size).encode()
        return strip_def

    def _rename_nodes(graph_def, rename_func):
        res_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = res_def.node.add()
            n.MergeFrom(n0)
            n.name = rename_func(n.name)
            for i, s in enumerate(n.input):
                n.input[i] = rename_func(s) if s[0] != '^' else '^' + rename_func(s[1:])
        return res_def

    def _show_entire_graph(graph_def, max_const_size=32):
        """Visualize TensorFlow graph."""
        if hasattr(graph_def, 'as_graph_def'):
            graph_def = graph_def.as_graph_def()
        strip_def = _strip_consts(graph_def, max_const_size=max_const_size)
        code = """
            <script>
              function load() {{
                document.getElementById("{id}").pbtxt = {data};
              }}
            </script>
            <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
            <div style="height:600px">
              <tf-graph-basic id="{id}"></tf-graph-basic>
            </div>
        """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

        iframe = """
            <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
        """.format(code.replace('"', '&quot;'))
        display(HTML(iframe))
    # Visualizing the network graph. Be sure expand the "mixed" nodes to see their
    # internal structure. We are going to visualize "Conv2D" nodes.
    tmp_def = _rename_nodes(graph_def, lambda s: "/".join(s.split('_', 1)))
    _show_entire_graph(tmp_def)
