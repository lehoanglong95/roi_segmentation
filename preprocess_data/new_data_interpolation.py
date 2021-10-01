import numpy as np
import os


if __name__ == '__main__':
    base_dir = "/home/longlh/hard_2/CT dicom ver 2.0"
    dirs = os.listdir(base_dir)
    for idx, d in enumerate(dirs):
        try:
            print(f"{idx+1} / {len(dirs)}")
            input = np.load(f"{base_dir}/{d}/input.npy")
            gt = np.load(f"{base_dir}/{d}/gt.npy")
            new_gt = np.zeros(gt.shape)
            l = []
            for idx in range(len(gt)):
                if len(np.unique(gt[idx])) > 1:
                    l.append(idx)
            # for e in l:
            #     new_gt[e] = gt[e]
            # start = min(l)
            # end = max(l)
            for idx in range(len(l) - 1):
                for e in range(l[idx], l[idx+1]):
                    new_gt[e] = gt[l[idx]]
            np.save(f"{base_dir}/{d}/new_gt.npy", new_gt)
        except:
            continue

