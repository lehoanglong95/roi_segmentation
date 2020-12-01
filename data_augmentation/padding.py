from utils.constant import TargetSize
import torch
import torch.nn.functional as F
import numpy as np

class Padding(object):

    def __init__(self, target_size):
        assert isinstance(target_size, TargetSize)
        self.target_size = target_size

    def __call__(self, items):
        c, h, w = items[0].shape
        # if h >= self.target_size.height and w >= self.target_size.width:
        #     return items
        # else:
        outputs = []
        for item in items:
            item = torch.from_numpy(item.astype(np.float32))
            if self.target_size.width >= w:
                w_diff = (self.target_size.width - w) // 2
                w_diff_plus = w_diff + 1 if (self.target_size.width - w) % 2 != 0 else w_diff
            else:
                w_diff = -((w - self.target_size.width) // 2)
                w_diff_plus = w_diff - 1 if (self.target_size.width - w) % 2 != 0 else w_diff
            if self.target_size.height >= h:
                h_diff = (self.target_size.height - h) // 2
                h_diff_plus = h_diff + 1 if (self.target_size.height - h) % 2 != 0 else h_diff
            else:
                h_diff = -((h - self.target_size.height) // 2)
                h_diff_plus = h_diff - 1 if (self.target_size.height - h) % 2 != 0 else h_diff
            output_item = F.pad(item, (w_diff, w_diff_plus, h_diff, h_diff_plus))
            outputs.append(output_item.numpy())
        return outputs

if __name__ == '__main__':
    import numpy as np
    a = [np.random.randint(0, 5, (2, 4, 4))]
    print(a)
    b = Padding(TargetSize(4, 4))(a)
    print(b)