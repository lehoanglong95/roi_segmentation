from utils.constant import TargetSize
import cv2
import numpy as np

class Padding(object):

    def __init__(self, target_size):
        assert isinstance(target_size, TargetSize)
        self.target_size = target_size

    def __call__(self, items):
        outputs = []
        c, h, w = items[0].shape
        for item in items:
            temp_item = []
            for img in item:
                new_img = cv2.resize(img, (self.target_size.height, self.target_size.width))
                temp_item.append(new_img)
                np.array()

if __name__ == '__main__':
    import numpy as np
    a = [np.random.randint(0, 5, (2, 4, 4))]
    print(a)
    b = Padding(TargetSize(4, 4))(a)
    print(b)