import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data

class MyDataset(data.Dataset):
    def __init__(self):
        self.data = torch.randn(10, 3, 24, 24)
        self.target = torch.randint(0, 10, (10,))

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        return {'data': x, 'target': y}

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    pass
    """
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

    '''
    for i in range(len(mask_array)):
        for j in range(len(mask_array[i])):
            print(mask_array[i,j], end=' ')
        print()
    index = torch.LongTensor([[1, 2, 4, 5]])
    '''

    #print(masks(index))

    tensor = torch.randn((3,64))
    tensor = tensor.unsqueeze(dim=1)
    #print(tensor.size())
    tensor = tensor.expand((3,3,64))
    #print(tensor)

    index = Variable(torch.LongTensor(range(8)))

    index = index.unsqueeze(dim=0)  # 1, num_conditions
    index = index.expand((3, n_conditions))  # batch_size, num_conditions

    print(index)
    print(index.size())

    result = masks(index)
    print(result)
    print(result.size())
    """
    """
    embed_feature = torch.randn((3, 8, 10))
    attention_weight = torch.randn((3, 8))
    attention_weight = attention_weight.unsqueeze(dim=2)
    attention_weight = attention_weight.expand((3, 8, 10))

    print(attention_weight)
    embed_feature * attention_weight
    """

    """
    example = torch.randn((3, 4, 3))
    #print(example)
    result = torch.sum(example, dim=1)

    print(example)
    print(result)
    print('example : ', example.size())
    print('result : ', result.size())
    """

    """
    e = torch.zeros((10))
    e[3] = 1
    #print(e)

    y = torch.ones_like(e) * -1
    print(y.sign())

    dataset = MyDataset()
    loader = data.DataLoader(
        dataset,
        batch_size=2,
        num_workers=2
    )

    for batch in loader:
        data = batch['data']
        target = batch['target']
        print(data.shape)
        print(target.shape)
    """
    """
    a_list = [[1,1],[2,2],[3,3],[4,4]]

    for idx, data in enumerate(a_list):
        a, b = data
        print('{} {} {}', idx, a, b)
    """

    """
    example = torch.zeros((10))
    example[5] = 1
    print(example)

    concat_example = torch.cat((example, example), 0)
    print(concat_example.size())
    
    """

    """
    a = torch.randn((2, 10))
    a_1, a_2 = torch.chunk(a,2,dim=0)
    b = torch.randn((2, 10))
    b_1, b_2 = torch.chunk(b, 2, dim=0)
    print(a)
    print(b)
    print(torch.mean((a - b)**2, dim=1))
    print(torch.mean((a_1 - b_1)**2, dim=1))
    print(torch.mean((a_2 - b_2)**2, dim=1))
    """

    b = torch.ones((4,1))

    b = b / 2.0

    print(b)

