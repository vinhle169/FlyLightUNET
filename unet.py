import torch
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F


# Building block unit of encoder and decoder architecture
class Block(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = Conv2d(inChannels, outChannels, 3)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, 3)

    def forward(self, inp):
        # Applies the ordering we have set above which is conv1 -> relu -> conv2
        return self.conv2(self.relu(self.conv1(inp)))


class Encoder(Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        super().__init__()
        # Stores our encoder blocks which are supposed to overtime increase channel size
        self.encoder_blocks = ModuleList([
            Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)
        ])
        # Reduces spatial dimensions by factor of 2
        self.pool = MaxPool2d(2)

    def forward(self, inp):
        block_outputs = []
        for block in self.encoder_blocks:
            # pass input through encoder block
            inp = block(inp)
            # store output
            block_outputs.append(inp)
            # apply maxpooling to output to pass on to next block
            inp = self.pool(inp)
        return block_outputs


class Decoder(Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        self.channels = channels
        # up-sampler block
        self.up_convs = ModuleList([
            ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)
        ])
        # down-sampler block
        self.dec_blocks = ModuleList([
            Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)
        ])

    def forward(self, inp, enc_features):
        for i in range(len(self.channels) - 1):
            # upsample
            inp = self.up_convs[i][inp]
            # crop features and concatenate with upsampled features
            enc_feat = self.crop(enc_features[i], inp)
            inp = torch.cat([inp, enc_feat], dim=1)
            # pass through decoder block
            inp = self.dec_blocks[i][inp]
        return inp

    def crop(self, enc_features, inp):
        # grab dims of inputs then crop encoder
        _, _, h, w = inp.shape
        enc_features = CenterCrop([h, w])(enc_features)
        return enc_features
