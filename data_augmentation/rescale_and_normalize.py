import numpy as np


class RescaleAndNormalize(object):

    def __call__(self, items):
        # last ele in items is labels with only 2 values 0 and 1. So we dont rescale and normalize
        outputs = []
        for i in range(0, len(items) - 1):
            if i == 0 or i == 2:
                rescaled = items[i] / np.max(items[i])
                mean = np.mean(rescaled)
                std = np.std(rescaled)
                outputs.append(np.float32((rescaled - mean) / std))
            else:
                outputs.append(items[i])
        outputs.append(items[-1])
        return outputs

# if __name__ == '__main__':
    # a = [np.random.randint(0, 5, (2, 2, 2)), np.random.randint(0, 5, (2, 2, 2))]
    # b = RescaleAndNormalize()(a)
    # print(b)