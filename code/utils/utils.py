import os, sys, math, time, inspect, cPickle
import numpy as np


def dump_to_pkl(x,fname):
    f = file(fname, 'wb')
    cPickle.dump(x, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def unpickle(path):
    ''' For cifar-10 data, it will return dictionary'''
    #Load the cifar 10
    f = open(path, 'rb')
    data = cPickle.load(f)
    f.close()
    return data 


def decode_seq(script, dic):
    """Returns decription of the script (sequence) based on dictionary dic"""

    decription = []
    ##  Convert text to indice
    for char in script:

        if isinstance(char, type(np.asarray([0]))) : char = char[0]
        ind = np.asarray([dic[char]])
        decription.append(ind)
    decription = np.asarray(decription).T
    return decription


def class_vars(obj):
    """Code from https://github.com/devsisters/DQN-tensorflow/blob/master/dqn/base.py"""
    return {k:v for k, v in inspect.getmembers(obj) \
        if not k.startswith('__') and not callable(k)}


def base_name(var):
    """Extracts value passed to name= when creating a variable
    
    Code from https://github.com/devsisters/DQN-tensorflow/blob/master/dqn/__init__.py
    """
    return var.name.split('/')[-1].split(':')[0]


def build_submission(filename, file_list):
    """Helper utility to check homework assignment submissions and package them.

    Parameters
    ----------
    filename : str
        Output zip file name
    file_list : tuple
        Tuple of files to include
    """
    # check each file exists
    for part_i, file_i in enumerate(file_list):
        assert os.path.exists(file_i), \
            '\nYou are missing the file {}.  '.format(file_i) + \
            'It does not look like you have completed Part {}.'.format(
                part_i + 1)

    # great, each file exists
    print('It looks like you have completed each part!')

    def zipdir(path, zf):
        for root, dirs, files in os.walk(path):
            for file in files:
                # make sure the files are part of the necessary file list
                if file.endswith(file_list):
                    zf.write(os.path.join(root, file))

    # create a zip file with the necessary files
    zipf = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED)
    zipdir('.', zipf)
    zipf.close()
    print('Great job!!!')
    print('Now submit the file:\n{}\nto Kadenze for grading!'.format(
        os.path.abspath(filename)))



def download_and_extract_tar(path, dst):
    """Download and extract a tar file.

    Parameters
    ----------
    path : str
        Url to tar file to download.
    dst : str
        Location to save tar file contents.
    """
    import tarfile
    filepath = download(path)
    if not os.path.exists(dst):
        os.makedirs(dst)
        tarfile.open(filepath, 'r:gz').extractall(dst)


def download_and_extract_zip(path, dst):
    """Download and extract a zip file.

    Parameters
    ----------
    path : str
        Url to zip file to download.
    dst : str
        Location to save zip file contents.
    """
    import zipfile
    filepath = download(path)
    if not os.path.exists(dst):
        os.makedirs(dst)
        zf = zipfile.ZipFile(file=filepath)
        zf.extractall(dst)

def download(path):
    """Use urllib to download a file.

    Parameters
    ----------
    path : str
        Url to download

    Returns
    -------
    path : str
        Location of downloaded file.
    """
    import os
    from six.moves import urllib

    fname = path.split('/')[-1]
    if os.path.exists(fname):
        return fname

    print('Downloading ' + path)

    def progress(count, block_size, total_size):
        if count % 20 == 0:
            print('Downloaded %02.02f/%02.02f MB' % (
                count * block_size / 1024.0 / 1024.0,
                total_size / 1024.0 / 1024.0))

    filepath, _ = urllib.request.urlretrieve(
        path, filename=fname, reporthook=progress)
    return filepath


