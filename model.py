import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models


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
        self.num_conditions = config['TRAINING_CONFIG']['NUM_CONDITION']
        self.num_category = config['TRAINING_CONFIG']['NUM_CATE']
        self.embedding_size = config['TRAINING_CONFIG']['EMD_DIM']
        self.pre_mask = config['TRAINING_CONFIG']['PRE_MASK'] == 'True'
        self.image_net = config['TRAINING_CONFIG']['IMAGE_NET'] == 'True'
        self.backbone_name = config['TRAINING_CONFIG']['BACKBONE']
        self.cate_net = list()
        self.cate_net.append(nn.Linear(self.num_category * 2, self.embedding_size))
        self.cate_net.append(nn.ReLU())
        self.cate_net.append(nn.Linear(self.embedding_size, self.embedding_size))
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

    def forward(self, image, image_category, concat_categories):
        feature_x = self.backbone(image)
        self.mask = self.masks(image_category)
        if self.learnedmask:
            self.mask = torch.nn.functional.relu(self.mask)

        masked_embedding = feature_x * self.mask
        norm = torch.norm(masked_embedding, p=2, dim=1) + 1e-10
        masked_embedding = masked_embedding / norm.expand_as(masked_embedding)
        return masked_embedding, self.mask.norm(1), feature_x.norm(2), masked_embedding.norm(2)

