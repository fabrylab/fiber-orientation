import numpy as np
import os

def flatten_dict(*args):
    ret_list = [i for i in range(len(args))]
    for i, ar in enumerate(args):
        ret_list[i] = np.concatenate([*ar.values()], axis=0)
    return ret_list

def createFolder(directory):
    '''
    function to create directories if they dont already exist
    '''
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    except OSError:
        print('Error: Creating directory. ' + directory)


def make_iterable(value):
    if not hasattr(value, '__iter__') or isinstance(value,str):
        return [value]
    else:
        return value
