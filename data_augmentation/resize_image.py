from utils.constant import TargetSize
import cv2
import numpy as np

class ResizeImage(object):

    def __init__(self, target_size):
        assert isinstance(target_size, TargetSize)
        self.target_size = target_size

    def __call__(self, items):
        outputs = []
        for item in items:
            temp_item = []
            for img in item:
                new_img = cv2.resize(img, (self.target_size.height, self.target_size.width))
                new_img = new_img.astype(np.float32)
                temp_item.append(new_img)
            temp_item = np.array(temp_item)
            outputs.append(temp_item)
        return outputs


if __name__ == '__main__':
    import numpy as np
    a = [np.random.randint(0, 5, (2, 4, 4))]
    print(a)
    b = Padding(TargetSize(4, 4))(a)
    print(b)