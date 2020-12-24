import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models

from torch.autograd import Variable


class ConditionalSimNet(nn.Module):
    def __init__(self, config):
        """ embeddingnet: The network that projects the inputs into an embedding of embedding_size
            n_conditions: Integer defining number of different similarity notions
            embedding_size: Number of dimensions of the embedding output from the embeddingnet
            learnedmask: Boolean indicating whether masks are learned or fixed
            prein: Boolean indicating whether masks are initialized in equally sized disjoint
                sections or random otherwise"""
        # embeddingnet, n_conditions, embedding_size, learnedmask=True, prein=False
        super(ConditionalSimNet, self).__init__()
        self.learnedmask = config['TRAINING_CONFIG']['MASK_LEARN'] == 'True'
        self.num_conditions = config['TRAINING_CONFIG']['NUM_CONDITIONS'] # num_masks
        self.num_category = config['TRAINING_CONFIG']['NUM_CATE']
        self.embedding_size = config['TRAINING_CONFIG']['EMD_DIM']
        self.pre_mask = config['TRAINING_CONFIG']['PRE_MASK'] == 'True'
        self.image_net = config['TRAINING_CONFIG']['IMAGE_NET'] == 'True'
        self.backbone_name = config['TRAINING_CONFIG']['BACKBONE']
        self.cate_net = list()
        self.cate_net.append(nn.Linear(self.num_category * 2, self.num_conditions))
        self.cate_net.append(nn.ReLU(inplace=True))
        self.cate_net.append(nn.Linear(self.num_conditions, self.num_conditions))
        self.cate_net.append(nn.Softmax(dim=1))
        self.cate_net = nn.Sequential(*self.cate_net)

        assert self.backbone_name in ['resnet18', 'resnet50']
        if self.backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=self.image_net)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, self.embedding_size)
        elif self.backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=self.image_net)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, self.embedding_size)

        # create the mask
        if self.learnedmask:
            if self.pre_mask:
                # define masks
                self.masks = torch.nn.Embedding(self.num_conditions, self.embedding_size)
                # initialize masks
                mask_array = np.zeros([self.num_conditions, self.embedding_size])
                mask_array.fill(0.1)
                mask_len = int(self.embedding_size / self.num_conditions)
                for i in range(self.num_conditions):
                    mask_array[i, i*mask_len:(i+1)*mask_len] = 1
                # no gradients for the masks
                self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=True)
            else:
                # define masks with gradients
                self.masks = torch.nn.Embedding(self.num_conditions, self.embedding_size)
                # initialize weights
                self.masks.weight.data.normal_(0.9, 0.7) # 0.1, 0.005
        else:
            # define masks
            self.masks = torch.nn.Embedding(self.num_conditions, self.embedding_size)
            # initialize masks
            mask_array = np.zeros([self.num_conditions, self.embedding_size])
            mask_len = int(self.embedding_size / self.num_conditions)
            for i in range(self.num_conditions):
                mask_array[i, i*mask_len:(i+1)*mask_len] = 1
            # no gradients for the masks
            self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=False)

    def forward(self, image, concat_categories):
        # image (B,C,H,W)
        # image_category (B, NUM_CATE)
        # concat_categories (B, NUM_CATE * @)

        feature_x = self.backbone(image) # Batch, embedding_dims
        feature_x = feature_x.unsqueeze(dim=1) # Batch, 1, embedding_dims
        b, _, _ = feature_x.size()
        feature_x = feature_x.expand((b, self.num_conditions, self.embedding_size))  # Batch, num_conditions, embedding_dims

        index = Variable(torch.LongTensor(range(self.num_conditions)))

        if image.is_cuda:
            index = index.cuda() # num_conditions
        index = index.unsqueeze(dim=0) # 1, num_conditions
        index = index.expand((b, self.num_conditions)) # batch_size, num_conditions

        embed = self.masks(index) # batch_size, num_conditions, embedding_dims
        embed_feature = embed * feature_x # batch_size, num_conditions, embedding_dims

        attention_weight = self.cate_net(concat_categories) # batch_size, num_conditions
        attention_weight = attention_weight.unsqueeze(dim=2)
        attention_weight = attention_weight.expand((b, self.num_conditions, self.embedding_size)) # batch_size, num_conditions, embedding_dims

        weighted_feature = embed_feature * attention_weight # batch_size, num_conditions, embedding_dims

        final_feature = torch.sum(weighted_feature, dim=1) # batch_size, embedding_dims

        return final_feature

