import json
import torch
import random
import os.path as osp

from torch.utils import data
from torchvision import transforms as T
from PIL import Image


class DataSet(data.Dataset):

    def __init__(self, config, img_transform):
        self.img_transform = img_transform
        self.num_conditions = config['TRAINING_CONFIG']['NUM_CONDITIONS']
        self.num_outfit = config['TRAINING_CONFIG']['NUM_OUTFIT']
        self.num_negative = config['TRAINING_CONFIG']['NUM_NEGATIVE']
        self.data_dir = config['TRAINING_CONFIG']['DATA_DIR']
        if config['TRAINING_CONFIG']['MODE'] == 'train':
            self.json_path = config['TRAINING_CONFIG']['TRAIN_JSON']
        elif config['TRAINING_CONFIG']['MODE'] == 'testing':
            self.json_path = config['TRAINING_CONFIG']['TESTING_JSON']
        elif config['TRAINING_CONFIG']['MODE'] == 'indexing':
            self.json_path = config['TRAINING_CONFIG']['INDEXING_JSON']

        with open(self.json_path, 'r') as fp:
            self.data_arr = json.load(fp) # list [
                                          # [positive_path, cate_num],
                                          # [[each_outfit image path, cate]...],
                                          # [[negative image path, cate]...]
                                          # ]

    def __getitem__(self, index):

        positive, outfit_list, negative_list = self.data_arr[index]

        if random.random() > 0.5 and self.get_negative_changing(positive, negative_list):
            loss_tensor = torch.ones(1) * -1
        else:
            loss_tensor = torch.ones(1)

        positive_path, positive_cate = positive
        positive_image = self.img_transform(Image.open(osp.join(self.data_dir, positive_path)))
        positive_onehot = torch.zeros((self.num_conditions), dtype=torch.long)
        positive_onehot[positive_cate] = 1

        data_dict = {
            'positive_image' : positive_image,
            'positive_cate' : torch.LongTensor(positive_cate),
            'loss_tensor' : loss_tensor,
        }

        self.get_list_data(data_dict, outfit_list, positive_onehot, prefix='outfit_{}_{}')
        self.get_list_data(data_dict, negative_list, positive_onehot, prefix='negative_{}_{}')

        return data_dict

    def __len__(self):
        """Return the number of images."""
        return len(self.data_arr)

    def get_list_data(self, data_dict, data_list, positive_onehot, prefix='outfit_{}_{}'):

        for idx, data in enumerate(data_list):
            image_path, cate = data
            image = self.img_transform(Image.open(osp.join(self.data_dir, image_path)))
            one_hot = torch.zeros((self.num_conditions), dtype=torch.long)
            one_hot[cate] = 1
            data_dict[prefix.format('image', idx)] = image
            data_dict[prefix.format('onehot', idx)] = torch.cat((positive_onehot, one_hot),dim=0)

    def get_negative_changing(self, positive, negative_list):

        positive_path, positive_cate = positive

        for idx, data in enumerate(negative_list):
            image_path, cate = data
            if cate == positive_cate:
                positive[0] = positive_path
                positive[1] = positive_cate
                return True

        return False


def get_loader(config):

    img_transform = list()
    img_size = config['MODEL_CONFIG']['IMG_SIZE']

    img_transform.append(T.Resize((img_size, img_size)))
    img_transform.append(T.ToTensor())
    img_transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    img_transform = T.Compose(img_transform)

    if config['TRAINING_CONFIG']['MODE'] == 'train':
        batch_size = config['TRAINING_CONFIG']['BATCH_SIZE']
    else:
        batch_size = 1

    dataset = DataSet(config, img_transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(config['TRAINING_CONFIG']['MODE'] == 'train'),
                                  num_workers=config['TRAINING_CONFIG']['NUM_WORKER'],
                                  drop_last=True)
    return data_loader
