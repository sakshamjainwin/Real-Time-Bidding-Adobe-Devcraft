import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

def weight_init(layers):
    """
    Initialize the weights of the layers.
    :param layers: A list of layers.
    """
    for layer in layers:
        if isinstance(layer, nn.BatchNorm1d):
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()
        elif isinstance(layer, nn.Linear):
            n = layer.in_features
            y = 1.0 / np.sqrt(n)
            layer.weight.data.uniform_(-y, y)
            layer.bias.data.fill_(0)
            # nn.init.kaiming_normal_(layer.weight.data, nonlinearity='relu')

class OuterPNN(nn.Module):
    """
    Outer Product-based Neural Network (OuterPNN) model.
    """
    def __init__(self,
                 feature_nums,
                 field_nums,
                 latent_dims,
                 output_dim=1):
        """
        Initialize the OuterPNN model.

        :param feature_nums: The number of features.
        :param field_nums: The number of fields.
        :param latent_dims: The dimensionality of the latent space.
        :param output_dim: The dimensionality of the output. Default: 1.
        """
        super(OuterPNN, self).__init__()
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.latent_dims = latent_dims

        self.feature_embedding = nn.Embedding(self.feature_nums, self.latent_dims)
        nn.init.xavier_uniform_(self.feature_embedding.weight.data)

        deep_input_dims = self.latent_dims + self.field_nums * self.latent_dims
        layers = list()

        neuron_nums = [300, 300, 300]
        for neuron_num in neuron_nums:
            layers.append(nn.Linear(deep_input_dims, neuron_num))
            # layers.append(nn.BatchNorm1d(neuron_num))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            deep_input_dims = neuron_num

        layers.append(nn.Linear(deep_input_dims, 1))

        weight_init(layers)

        self.mlp = nn.Sequential(*layers)

        # kernel for outer product transformation
        self.kernel = torch.ones((self.latent_dims, self.latent_dims)).cuda()
        # nn.init.normal_(self.kernel)

    def forward(self, x):
        """
        Forward pass of the OuterPNN model.

        :param x: Int tensor of size (batch_size, feature_nums)
        :return: pctrs: Predicted CTRs, tensor of size (batch_size, 1).
        """
        embedding_x = self.feature_embedding(x)

        sum_embedding_x = torch.sum(embedding_x, dim=1).unsqueeze(1) # sum embeddings of all fields
        outer_product = torch.mul(torch.mul(sum_embedding_x, self.kernel), sum_embedding_x)

        cross_item = torch.sum(outer_product, dim=1)  # reduce dimensionality

        cat_x = torch.cat([embedding_x.view(-1, self.field_nums * self.latent_dims), cross_item], dim=1)
        out = self.mlp(cat_x)

        return torch.sigmoid(out)
    