import numpy
import os
import sys

path = '../train_node_real40'

fns = os.listdir(path)
# print(fns)
for fn in fns:
    if '.npy' in fn.split('_')[3]:
        # np90 = numpy.rot90(numpy.load(os.path.join(path, fn)), k = 1, axes= (0, 1))
        # numpy.save(os.path.join(path, fn.split('.')[0] + '_90.npy'), np90)

        # np180 = numpy.rot90(np90 , k = 1, axes= (0, 1))
        # numpy.save(os.path.join(path, fn.split('.')[0] + '_180.npy'), np180)

        # np270 = numpy.rot90(np90 , k = 2, axes= (0, 1))
        # numpy.save(os.path.join(path, fn.split('.')[0] + '_270.npy'), np270)

        # np_flip_0 = numpy.load(os.path.join(path, fn)).transpose((1, 0, 2))
        # numpy.save(os.path.join(path, fn.split('.')[0] + '_flip_0.npy'), np_flip_0)
        
        # np_flip_90 = numpy.rot90(np_flip_0, k= 1, axes= (0, 1))
        # numpy.save(os.path.join(path, fn.split('.')[0] + '_flip_90.npy'), np_flip_90)
        
        # np_flip_180 = numpy.rot90(np_flip_0, k= 2, axes= (0, 1))
        # numpy.save(os.path.join(path, fn.split('.')[0] + '_flip_180.npy'), np_flip_180)
        
        # np_flip_270 = numpy.rot90(np_flip_0, k= 3, axes= (0, 1))
        # numpy.save(os.path.join(path, fn.split('.')[0] + '_flip_270.npy'), np_flip_270)

        np_around = numpy.rot90(numpy.load(os.path.join(path, fn)), k = 2, axes= (1, 2))
        numpy.save(os.path.join(path, fn.split('.')[0] + '_around.npy'), np_around)

        np_around_90 = numpy.rot90(np_around , k = 1, axes= (0, 1))
        numpy.save(os.path.join(path, fn.split('.')[0] + '_around90.npy'), np_around_90)

        np_around_180 = numpy.rot90(np_around , k = 2, axes= (0, 1))
        numpy.save(os.path.join(path, fn.split('.')[0] + '_around180.npy'), np_around_180)

        np_around_270 = numpy.rot90(np_around, k = 3, axes = (0, 1))
        numpy.save(os.path.join(path, fn.split('.')[0] + '_around270.npy'), np_around_270)
        
        np_around_flip = np_around.transpose((1, 0, 2))
        numpy.save(os.path.join(path, fn.split('.')[0] + '_around_flip.npy'), np_around_flip)
        
        np_around_flip_90 = numpy.rot90(np_around_flip, k= 1, axes= (0, 1))
        numpy.save(os.path.join(path, fn.split('.')[0] + '_around_filp90.npy'), np_around_flip_90)
        
        np_around_flip_180 = numpy.rot90(np_around_flip, k= 2, axes= (0, 1))
        numpy.save(os.path.join(path, fn.split('.')[0] + '_around_filp180.npy'), np_around_flip_180)

        np_around_flip_270 = numpy.rot90(np_around_flip, k= 3, axes= (0, 1))
        numpy.save(os.path.join(path, fn.split('.')[0] + '_around_filp270.npy'), np_around_flip_270)
