import numpy as np
import os
import matplotlib
from skimage.util import montage
import pylab as plt


def show_img(im, gt=None, show=False, save_dir=None, mask=None, suffix=''):
    grid_shape = 4, np.ceil(im.shape[0] / 4)
    im = montage(im, grid_shape=grid_shape)
    if gt is not None:
        markers = np.zeros_like(gt)
        for i, marker in enumerate(markers):
            if gt[i].sum() > 0:
                marker[10:-10, 10:-10] += 1

        gt = montage(gt, grid_shape=grid_shape)
        mask = montage(mask, grid_shape=grid_shape) if mask is not None else mask
        markers = montage(markers, grid_shape=grid_shape)
    matplotlib.use('TkAgg')
    plt.figure(figsize=(16, 8))
    plt.imshow(im, cmap='gray')
    if gt is not None:
        plt.contour(gt, linewidths=.3, colors='y')
    plt.axis('off')
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    if show:
        plt.show()
    elif save_dir is not None:
        plt.savefig(f"{save_dir}/{suffix}")
        plt.close()


def show_pred(im, pred, gt=None, show=False, save_dir=None, mask=None, suffix=''):
    grid_shape = 5, np.ceil(im.shape[0] / 5)
    im = montage(im, grid_shape=grid_shape)
    print(im.shape)
    if gt is not None:
        markers = np.zeros_like(gt)
        for i, marker in enumerate(markers):
            if gt[i].sum() > 0:
                marker[10:-10, 10:-10] += 1

        gt = montage(gt, grid_shape=grid_shape)
        pred = montage(pred, grid_shape=grid_shape)
        mask = montage(mask, grid_shape=grid_shape) if mask is not None else mask
        markers = montage(markers, grid_shape=grid_shape)
    if show:
        matplotlib.use('TkAgg')
        hw_ratio = im.shape[0] / im.shape[1]
        fig_sz = 16
        fig = plt.figure(figsize=(fig_sz, fig_sz * hw_ratio))
        plt.subplots_adjust(0, 0, 1, 1)
        plt.imshow(im, cmap='gray')
        plt.contour(pred, linewidths=.3, colors='r')
        if gt is not None:
            plt.contour(gt, linewidths=.3, colors='y')
        plt.axis('off')
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        plt.savefig(f"{save_dir}/{suffix}")
        plt.close()
        # plt.show()

def save_img(d):
    print(d)
    adc = np.load(f"{base_dir}/{d}/new_input.npy")[0]
    # dwi = np.load(f"{base_dir}/{d}/DWI/sampled_input.npy")
    pred = np.load(f"{base_dir}/{d}/predict.npy")
    gt = np.load(f"{base_dir}/{d}/new_gt.npy")
    # print(adc.shape)
    # print(pred.shape)
    # print(gt.shape)
    show_pred(adc, pred=pred, gt=gt, save_dir=f"{base_dir}/{d}", suffix="dwi_gt_pred", show=True)
    # show_pred(dwi, gt=None, save_dir=f"{base_dir}/{d}/ADC", suffix="dwi_without_gt")

# base_dir = "/home/longlh/hard_2/roi_numpy"
# base_dir = "/home/longlh/hard_2/PycharmProjects/roi_segmentation/result_relabel"
base_dir = "/home/longlh/hard_2/roi_numpy"

if __name__ == '__main__':
    save_img("BP908 20181120/ADC")
    # import os
    # import multiprocessing
    # pool = multiprocessing.Pool(8)
    # dirs = os.listdir(base_dir)
    # pool.map(save_img, dirs)
    # pool.close()
