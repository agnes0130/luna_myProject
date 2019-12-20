import os
import numpy
path = '../train_node_real'
for fn in os.listdir(path):
    shape = numpy.load(os.path.join(path, fn)).shape
    if shape != (64, 64, 6):
        os.remove(os.path.join(path,fn))
        print('!error: ', fn)