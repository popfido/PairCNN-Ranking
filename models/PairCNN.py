# coding=utf-8
from .BasicModule import BasicModule
from torch import nn
import torch
from torch.autograd import Variable


class PairCNN(BasicModule):
    def __init__(self, opt):
        super(PairCNN, self).__init__()
        self.model_name = "paircnn"

        self.embed_left = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.embed_right = nn.Embedding(opt.vocab_size, opt.embedding_dim)

        self.pooled_layer_left = []
        self.pooled_layer_right = []
        filter_sizes = [int(size) for size in opt.filter_sizes.split(',')]
        for i, filter_size in enumerate(filter_sizes):
            filter_shape = [filter_size, opt.embedding_dim, 1, opt.num_filters]
            left_layer = nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=opt.num_filters,
                          kernel_size=(filter_size, opt.embedding_dim),
                          stride=1,
                          padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(opt.max_len_left - filter_size + 1, 1),
                             stride=(1, 1),
                             padding=0
                             )
            )

            right_layer = nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=opt.num_filters,
                          kernel_size=(filter_size, opt.embedding_dim),
                          stride=1,
                          padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(opt.max_len_right - filter_size + 1, 1),
                             stride=(1, 1),
                             padding=0
                             )
            )

            if opt.use_gpu:
                left_layer = left_layer.cuda()
                right_layer = right_layer.cuda()

            self.pooled_layer_left.append(left_layer)
            self.pooled_layer_right.append(right_layer)

        self.num_filters_total = opt.num_filters * len(filter_sizes)

        self.similarity_layer = nn.Bilinear(self.num_filters_total, self.num_filters_total, 1)

        self.hidden_layer = nn.Sequential(
            nn.Linear(in_features=2 * self.num_filters_total + 1,
                      out_features=opt.num_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=1.0 - opt.dropout_keep_prob,
                       inplace=True),
            nn.Linear(in_features=opt.num_hidden,
                      out_features=2)
        )

        if opt.use_gpu:
            self.embed_left, self.embed_right = self.embed_left.cuda(), self.embed_right.cuda()
            self.similarity_layer = self.similarity_layer.cuda()
            self.hidden_layer = self.hidden_layer.cuda()

    def forward(self, input_data):
        assert type(input_data) is tuple

        embedded_left = torch.unsqueeze(self.embed_left(input_data[0]), 1)
        embedded_right = torch.unsqueeze(self.embed_right(input_data[1]), 1)

        pooled_outputs_left = [layer(embedded_left) for layer in self.pooled_layer_left]
        pooled_outputs_right = [layer(embedded_right) for layer in self.pooled_layer_right]

        h_pool_left = torch.cat(pooled_outputs_left, 3).view(-1, self.num_filters_total)
        h_pool_right = torch.cat(pooled_outputs_right, 3).view(-1, self.num_filters_total)
        similarity = self.similarity_layer(h_pool_left, h_pool_right)

        new_input = torch.cat([h_pool_left, similarity, h_pool_right], 1)

        scores = self.hidden_layer(new_input)
        values, predictions = torch.max(scores, 1)
        return scores, predictions
