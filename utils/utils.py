
import torch.nn as nn
import torch.nn.init as init
from scipy.ndimage import morphology
import numpy as np

def weight_init(module):
    '''
    initial params
    :param module: pytorch
    '''
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d) \
            or isinstance(module, nn.Linear):
        init.kaiming_uniform_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm3d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm1d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()


def write_to_file(content_list, path):
    '''
    write list
    :param content_list: save 
    :param path: data path
    '''
    with open(path, 'w') as f:
        for item in content_list:
            print(item, file=f)

def pixel_normailize(input, min_value, max_value):
    '''
    The normalization process is performed according to the input pixel value range.
    :param input: input data
    :param min_value: min pixel
    :param max_value: min pixel
    :return: processed data
    '''
    output = (input - min_value) / (max_value - min_value)
    output[output < 0] = 0
    output[output > 1] = 1

    return output

def create_distance_map(target_mask, sampling=1):
    input_1 = np.atleast_1d(target_mask.astype(np.bool))
    dta = morphology.distance_transform_edt(~input_1, sampling)
    #print(dta.max())
    # normalize
    dta /= 200.

    return 1 - np.clip(dta, 0, 1)

# set requies_grad=Fasle to avoid computation
def set_requires_grad(net, requires_grad=False):
    for param in net.parameters():
        param.requires_grad = requires_grad