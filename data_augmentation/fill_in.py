import numpy as np


class FillIn(object):

    def __init__(self, target_number_of_channels):
        self.target_number_of_channels = target_number_of_channels

    def __call__(self, items):
        outputs = []
        c, h, w = items[0].shape
        if c >= self.target_number_of_channels:
            return items
        else:
            for item in items:
                temp = np.zeros((self.target_number_of_channels - c, h, w))
                temp_items = np.append(item, temp, axis=0)
                outputs.append(temp_items.astype(np.float32))
            return outputs

if __name__ == '__main__':
    a = [np.random.randint(0, 5, (2, 2, 2)), np.random.randint(0, 5, (2, 2, 2))]
    print(a)
    b = FillIn(4)(a)
    print(b)