import os
import numpy as np
import pylab as plt
from skimage.util import montage
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from PySide2 import QtCore, QtGui
from util.misc import gen_bin_mask
from deep_learning_model import UNet
matplotlib.use('Qt5Agg')

from PySide2.QtWidgets import QWidget, QVBoxLayout
from util.register import ADC, DWI
from util.preprocess import register
import torch
from data_augmentation import Padding, RescaleAndNormalize, TargetSize

u_net = UNet(2, 2, 1, False, torch.device("cuda:0"), torch.device("cuda:1"))
u_net.load_state_dict(torch.load("/home/longlh/hard_2/PycharmProjects/roi_segmentation/"
                                   "pretrain/UNet_0.5DiceLoss_0.5FocalLoss_Adam_ADC_inputs_ADC_mask_DWI_inputs_DWI_mask_labels_2020-12-09-00-48-54_seed15/model_108.pth")["model_state_dict"])
padding = Padding(TargetSize(224, 224))
rescale = RescaleAndNormalize()

class DropCanvas(FigureCanvas):
    def __init__(self, parent=None, width=7, height=7, dpi=135):
        fig = Figure(figsize=(width, height),
                     dpi=dpi,
                     tight_layout={'h_pad': 0, 'w_pad': 0, 'pad': 0},
                     frameon=False,
                     facecolor='white',
                     edgecolor='white',
                     linewidth=1.)
        FigureCanvas.__init__(self, fig)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_axis_off()
        self.ca = None
        self.pth = None
        self.setParent(parent)
        self.setAcceptDrops(True)

    @staticmethod
    def adjust_path(pth):
        if os.name == 'nt':
            pth = pth.replace('file:///', '')
        else:
            pth = pth.replace('file://', '').rstrip('\r\n'). \
                replace('%20', ' ')
        return pth

    def dragEnterEvent(self, event):
        if not event.mimeData().hasText():
            event.ignore()
            return
        pth = self.adjust_path(event.mimeData().text())
        if os.path.isdir(pth):
            self.pth = pth
            event.accept()

    def dropEvent(self, event):
        if self.pth is None:
            return
        # print(self.pth)
        adc = ADC(self.pth)
        dwi = DWI(self.pth)
        # register(self.pth)
        # target = ADC(self.pth)
        # target.spacing = (3, 1, 1)
        #
        # adc_resampled = adc.resample(target)
        # dwi_resampled = dwi.resample(target)
        # adc = ADC(self.pth)
        # dwi = DWI(self.pth)
        # target = ADC(self.pth)
        # target.itkimage.SetSpacing([1, 1, 3])
        # adc_sampled = adc.resample(target)
        # dwi_sampled = dwi.resample(target)
        # data = register(self.pth)
        # adc = data["adc"]
        # dwi = data["dwi"]
        # cx, cy = adc.scan.shape[-1]//2 + np.random.choice(10), adc.scan.shape[-1]//2 + np.random.choice(10)
        # roi = np.tile(gen_bin_mask(adc.scan.shape[1:], 11, 11, cx, cy), (adc.scan.shape[0], 1, 1))
        # print(roi.shape)
        # n_rows = int(adc.scan.shape[0] ** (1 / 2))
        # n_cols = np.ceil(adc.scan.shape[0] / n_rows)
        # n = max(n_rows, n_cols)
        # print(f"ROW: {n_rows}")
        # print(f"COL: {n_cols}")
        concat_img = np.concatenate((adc.scan, dwi.scan), axis=0)
        concat_img.squeeze()

        # DEEP LEARNING
        # adc_img = padding([adc.scan])[0]
        adc_img = rescale(adc.scan)
        torch_adc_img = torch.from_numpy(adc_img.astype(np.float32))
        torch_adc_img = torch.unsqueeze(torch_adc_img, dim=0)
        torch_adc_img = torch.unsqueeze(torch_adc_img, dim=0)
        # dwi_img = padding([dwi.scan])[0]
        dwi_img = rescale(dwi.scan)
        torch_dwi_img = torch.from_numpy(dwi_img.astype(np.float32))
        torch_dwi_img = torch.unsqueeze(torch_dwi_img, dim=0)
        torch_dwi_img = torch.unsqueeze(torch_dwi_img, dim=0)
        torch_concat_img = torch.cat([torch_adc_img, torch_dwi_img], dim=1).to(torch.device("cuda:0"))
        # print(torch_concat_img.shape)
        # np.save("/home/longlh/hard_2/b.npy", torch_concat_img.squeeze().numpy())
        predict = u_net(torch_concat_img)
        print(torch.unique(predict))
        # print(predict.shape)
        # print(concat_img.shape)

        n_rows = int(concat_img.shape[0] ** (1 / 2))
        n_cols = np.ceil(concat_img.shape[0] / n_rows)
        n = max(n_rows, n_cols)
        img = montage(concat_img, grid_shape=(n, n), padding_width=0, fill=0)
        print(concat_img.shape)
        roi = torch.squeeze(predict).cpu().numpy()

        # dwi_img = montage(dwi.scan, grid_shape=(n, n), padding_width=0, fill=0)
        # print(img.shape)
        # print(dwi_img.shape)
        contour = montage(roi, grid_shape=(n, n), padding_width=0, fill=0)

        if self.ca is None:
            self.ax.clear()
            self.ax.set_axis_off()
            self.ca = self.ax.imshow(img, )
            self.cs = self.ax.contour(contour, colors='r', linewidths=.3)
        else:
            # self.reset_contours()
            self.ax.clear()
            self.ax.set_axis_off()
            self.ax.imshow(img)
            self.cs = self.ax.contour(contour, colors='r', linewidths=.3)

        self.ca.set_clim([0, img.max()])
        self.figure.canvas.draw()

    def drawRectangle(self, rect):
        # Draw the zoom rectangle to the QPainter.  _draw_rect_callback needs
        # to be called at the end of paintEvent.
        if rect is not None:
            def _draw_rect_callback(painter):
                # IN THIS EXAMPLE CHANGE BLACK FOR WHITE
                pen = QtGui.QPen(QtCore.Qt.white, 1 / self._dpi_ratio,
                                 QtCore.Qt.DashLine)
                painter.setPen(pen)
                painter.drawRect(*(pt / self._dpi_ratio for pt in rect))
        else:
            def _draw_rect_callback(painter):
                return
        self._draw_rect_callback = _draw_rect_callback
        self.update()

    def reset_contours(self):
        if isinstance(self.cs, list):
            return
        [c.remove() for c in self.cs.collections]
        self.cs = []


class WidgetPlot(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.setLayout(QVBoxLayout())
        self.canvas = DropCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background-color:Gray;")
        self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)
