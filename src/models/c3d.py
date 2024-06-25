# coding: utf-8
import torch
import torch.nn as nn


class C3D(nn.Module):
    """
    nb_classes: nb_classes in classification task, 101 for UCF101 dataset
    """

    def __init__(self, nb_classes, feature_layer):
        super(C3D, self).__init__()

        self.feature_layer = feature_layer

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        # self.fc8 = nn.Linear(4096, nb_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

    def forward_single(self, x):
        h = self.relu(self.conv1(x))
        h = self.pool1(h)
        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        out = h if self.feature_layer == 5 else None
        h = self.relu(self.fc6(h))
        out = h if self.feature_layer == 6 and out == None else out
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        out = h if self.feature_layer == 7 and out == None else out
        h = self.dropout(h)
        # logits = self.fc8(h) # for activity recognition
        return out  # for features extraction

    def forward(self, batch):
        if batch['frames'].dim() == 5:
            feat = self.forward_single(batch['frames'])
        return feat


def c3d_model(nb_classes, pretrainedpath, feature_layer):
    net = C3D(nb_classes=nb_classes, feature_layer=feature_layer)
    state_dict = torch.load(pretrainedpath)
    state_dict.pop('fc8.weight')
    state_dict.pop('fc8.bias')

    # new_dict = {}
    # for key, value in state_dict.items():
    #     new_key = key.replace('.conv.', '.')
    #     if new_key == 'conv1a.weight':
    #         new_key = 'conv1.weight'
    #     if new_key == 'conv1a.bias':
    #         new_key = 'conv1.bias'
    #     if new_key == 'conv2a.weight':
    #         new_key = 'conv2.weight'
    #     if new_key == 'conv2a.bias':
    #         new_key = 'conv2.bias'
    #     new_dict[new_key] = value
    # net.load_state_dict(new_dict)
    net.load_state_dict(state_dict)
    print("Received Pretrained model..")
    return net
