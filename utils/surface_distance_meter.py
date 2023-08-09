
import numpy as np
from scipy.ndimage import morphology


class SurfaceDistanceMeter(object):
    '''
    surface distance and hausdorf distance 
    refer to http://mlnotebook.github.io/post/surface-distance-function
    '''
    def __init__(self, input1, input2, sampling=1, connectivity=1):
        '''
        function
        :param input1,input2:（numpy）seg 
        :param sampling: pixel spacing
        :param connectivity: The connected domain size of the operator is used in the corrosion operation

        '''

        self.sds1, self.sds2 = self.surfd(input1, input2, sampling=sampling, connectivity=connectivity)

    def hd(self):
        '''
        hausdorf distance
        :return: HD
        '''
        return max(self.sds1.max(), self.sds2.max())

    def hd95(self):
        '''
        95%hausdorf
        :return: 95% hausdorf
        '''
        return np.percentile(np.hstack((self.sds1, self.sds2)), 95)

    def msd(self):
        '''
        calculate the average distance
        :return: average surface distance
        '''
        return np.mean(self.sds1.mean(), self.sds2.mean())

    def surfd(self, input1, input2, sampling=1, connectivity=1):
        '''
        :param input1,input2:（numpy）seg results
        :param sampling: spacing
        :param connectivity: The connected domain size of the operator is used in the corrosion operation
        :return: array，front-back ground distance
        '''
        #
        input_1 = np.atleast_1d(input1.astype(np.bool))
        input_2 = np.atleast_1d(input2.astype(np.bool))

        conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

        # S = input_1 - morphology.binary_erosion(input_1, conn)
        # Sprime = input_2 - morphology.binary_erosion(input_2, conn)
        S = input_1 ^ morphology.binary_erosion(input_1, conn)
        Sprime = input_2 ^ morphology.binary_erosion(input_2, conn)

        dta = morphology.distance_transform_edt(~S, sampling)
        dtb = morphology.distance_transform_edt(~Sprime, sampling)

        sds1 = np.ravel(dta[Sprime != 0])
        sds2 = np.ravel(dtb[S != 0])

        return sds1, sds2
    
    
