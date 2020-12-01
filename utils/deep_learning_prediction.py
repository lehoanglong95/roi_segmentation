import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

import sys


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


# if __name__ == "__main__":
#     img = np.load(
#         r'D:\workspace\Copied_from_C\Workspace\BrainKUCMC\April2019\2019 CMC-contrast extra\Ingenia CX (Philips)-Contrast-Normal\BP402 2019-01-05\PWI_DSC_Collateral\MatFiles\IMG_n01.npy')
#     model = Model()
#     model.dir_model = '../models'
#     model.predict(img)
