import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from torch.autograd import Variable

if __name__ == '__main__':
    pass
    n_conditions = 8
    embedding_size = 16
    mask_array = np.zeros([n_conditions, embedding_size])
    mask_array.fill(0.1)
    mask_len = int(embedding_size / n_conditions)
    #print(mask_len)
    for i in range(n_conditions):
        mask_array[i, i * mask_len:(i + 1) * mask_len] = 1

    masks = torch.nn.Embedding(n_conditions, embedding_size)
    masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=True)

    """
    for i in range(len(mask_array)):
        for j in range(len(mask_array[i])):
            print(mask_array[i,j], end=' ')
        print()
    index = torch.LongTensor([[1, 2, 4, 5]])
    """

    #print(masks(index))

    tensor = torch.randn((3,64))
    tensor = tensor.unsqueeze(dim=1)
    #print(tensor.size())
    tensor = tensor.expand((3,3,64))
    #print(tensor)

    index = Variable(torch.LongTensor(range(8)))

    print(index)
    print(masks(index))

    result = masks(index)
    print(result.size())


