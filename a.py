def show_img(im, gt=None, predict=None, show=False, save_dir=None, mask=None, suffix=''):
    grid_shape = 4, np.ceil(im.shape[0] / 4)
    im = montage(im, grid_shape=grid_shape)
    if gt is not None:
        markers = np.zeros_like(gt)
        for i, marker in enumerate(markers):
            if gt[i].sum() > 0:
                marker[10:-10, 10:-10] += 1

        gt = montage(gt, grid_shape=grid_shape)
        predict = montage(predict, grid_shape=grid_shape)
        mask = montage(mask, grid_shape=grid_shape) if mask is not None else mask
        markers = montage(markers, grid_shape=grid_shape)
    if show:
        matplotlib.use('TkAgg')
        plt.figure(figsize=(16, 8))
        plt.imshow(im, cmap='gray')
        if gt is not None:
            plt.contour(gt, linewidths=.3, colors='r')
            plt.contour(predict, linewidths=.3, colors='y')
        plt.axis('off')
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        plt.show()
    elif save_dir is not None:
        import cv2
        im_rgb = cv2.cvtColor((im / im.max() * 255).astype('uint8'), cv2.COLOR_GRAY2RGB)

        if gt is not None:
            # Draw VOIs
            cnt, hier = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im_rgb, cnt, -1, (255, 0, 0), 1)

            # Draw slide marker
            cnt, hier = cv2.findContours(markers, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im_rgb, cnt, -1, (0, 255, 255), 3)

        if mask is not None:
            # Draw VOIs
            cnt, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im_rgb, cnt, -1, (255, 255, 0), 1)

        plt.imsave(os.path.join(save_dir, f'plot_{suffix}.png'), im_rgb)

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
        #plt.show()
        plt.savefig(f"{save_dir}/{suffix}")