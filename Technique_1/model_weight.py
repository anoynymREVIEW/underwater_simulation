import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()        
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)

        print("size after interpolation")
        print(up_x.size())

        t_1 = torch.cat([up_x, concat_with], dim=1)
        print("size after concatenation")
        print(t_1.size())
        del up_x
        t_2 = self.convA(t_1)
        print("size after conv-A")
        print(t_2.size())
        del t_1
        t_3 = self.convB(t_2)
        print("size after conv-B")
        print(t_3.size())
        del t_2
        t_4 = self.leakyreluB(t_3)
        print("size after leaky Relu")
        print(t_4.size())
        del t_3
        return t_4


class Decoder(nn.Module):
    def __init__(self, num_features=1664, decoder_width = 1.0):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)

        self.up1 = UpSample(skip_input=features//1 + 256, output_features=features//2)
        self.up2 = UpSample(skip_input=features//2 + 128,  output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 + 64,  output_features=features//8)
        self.up4 = UpSample(skip_input=features//8 + 64,  output_features=features//16)

        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[12]
        x_d0 = self.conv2(F.relu(x_block4))

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        return self.conv3(x_d4)


class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out, kernel_size=1, stride=1, padding=1):
        super(ConvBNRelu, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class FlatAvgPool(nn.Module):

    def __init__(self):
        super(FlatAvgPool, self).__init__()

        self.flat_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.just_flatten = nn.Flatten()

    def forward(self, x):
        x_out = self.flat_avg_pool(x)
        x_out = self.just_flatten(x_out)

        return x_out


class FeatureLinearLayer(nn.Module):

    def __init__(self):
        super(FeatureLinearLayer, self).__init__()

        self.classifier = nn.Sequential(

            # FC 1
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            # FC 2
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            # FC 3
            nn.Dropout(p=0.5),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),

            # FC 4
            nn.Dropout(p=0.5),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),

            # FC 5
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.classifier(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.original_model = models.densenet169(pretrained=False)
        # num_ftrs = self.original_model.classifier.in_features
        # self.FCFeatures = nn.Linear(num_ftrs, 512)

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items():
            features.append(v(features[-1]))
        return features


class PTModel(nn.Module):
    def __init__(self):
        super(PTModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.Conv_1 = ConvBNRelu(1664, 512, 11, 1, 1)
        self.Conv_2 = ConvBNRelu(512, 256, 9, 1, 1)

        self.AvgPoolFlat = FlatAvgPool()
        self.Last_Linear_Layer = FeatureLinearLayer()
        self.SigmoidLayer = nn.Sigmoid()

    def forward(self, x):
        encode_part = self.encoder(x)
        last_block_feature = encode_part[12]
        
        print("size after last layer of encoder")
        print(last_block_feature.size())

        x_conv_out = self.Conv_1(last_block_feature.float())
        print("size after weight conv 1")
        print(x_conv_out.size())

        x_conv_out = self.Conv_2(x_conv_out)
        print("size after weight conv 2")
        print(x_conv_out.size())

        linear_features = self.AvgPoolFlat(x_conv_out)
        print("size after average pool flat")
        print(linear_features.size())

        weight_features = self.Last_Linear_Layer(linear_features)
        print("size after last linear layer")
        print(weight_features.size())
        
        weight_features = self.SigmoidLayer(weight_features)

        decode_part = self.decoder(encode_part)

        return decode_part, weight_features

