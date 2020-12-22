import json

from torch.utils import data
from torchvision import transforms as T
from PIL import Image


class DataSet(data.Dataset):

    def __init__(self, config, img_transform):
        self.img_transform = img_transform
        with open(config['TRAINING_CONFIG']['DATA_DIR'], 'r') as fp:
            self.data_arr = json.load(fp) # list [[positive_path, [outfit image path],[negative image path]]

    def __getitem__(self, index):

        positive_path, outfit_list, negative_list = self.data_arr[index]
        positive_image = self.img_transform(Image.open(positive_path))
        data_dict = {
            'positive_image' : positive_image ,
            'outfit_list' : self.get_list_data(outfit_list),
            'negative_list' : self.get_list_data(negative_list),
        }
        return data_dict


    def __len__(self):
        """Return the number of images."""
        return len(self.data_arr)

    def get_list_data(self, path_list):

        image_list = list()
        for image_path in path_list:
            image = self.img_transform(Image.open(image_path))
            image_list.append(image)

        return image_list


def get_loader(config):

    img_transform = list()
    img_size = config['MODEL_CONFIG']['IMG_SIZE']

    img_transform.append(T.Resize((img_size, img_size)))
    img_transform.append(T.ToTensor())
    img_transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    img_transform = T.Compose(img_transform)

    dataset = DataSet(config, img_transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['TRAINING_CONFIG']['BATCH_SIZE'],
                                  shuffle=(config['TRAINING_CONFIG']['MODE'] == 'train'),
                                  num_workers=config['TRAINING_CONFIG']['NUM_WORKER'],
                                  drop_last=True)
    return data_loader
