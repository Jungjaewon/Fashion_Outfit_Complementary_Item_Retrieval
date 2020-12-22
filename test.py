import numpy as np
import torchvision.models as models

if __name__ == '__main__':

    n_conditions = 8
    embedding_size = 64
    mask_array = np.zeros([n_conditions, embedding_size])
    mask_array.fill(0.1)
    mask_len = int(embedding_size / n_conditions)
    for i in range(n_conditions):
        mask_array[i, i * mask_len:(i + 1) * mask_len] = 1


    for i in range(len(mask_array)):
        for j in range(len(mask_array[i])):
            print(mask_array[i,j], end=' ')
        print()

    models.resnet50()