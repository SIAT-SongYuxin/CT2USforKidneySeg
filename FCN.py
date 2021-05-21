from torchvision import models
from torch import nn
import torch
import torch.nn.functional as F

class FCN_ResNet18(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        base_model = models.resnet18(pretrained=False)

        self.layers = list(base_model.children())
        layers=self.layers
        self.layer1 = nn.Sequential(*layers[:5])  # size=(N, 64, x.H/2, x.W/2)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.layer2 = layers[5]  # size=(N, 128, x.H/4, x.W/4)
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.layer3 = layers[6]  # size=(N, 256, x.H/8, x.W/8)
        self.upsample3 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.layer4 = layers[7]  # size=(N, 512, x.H/16, x.W/16)
        self.upsample4 = nn.Upsample(scale_factor=32, mode='bilinear')

        self.conv1k = nn.Conv2d(64 + 128 + 256 + 512, n_class, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        up1 = self.upsample1(x)
        x = self.layer2(x)
        up2 = self.upsample2(x)
        x = self.layer3(x)
        up3 = self.upsample3(x)
        x = self.layer4(x)
        up4 = self.upsample4(x)

        merge = torch.cat([up1, up2, up3, up4], dim=1)
        merge = self.conv1k(merge)
        #out = self.sigmoid(merge)
        out=merge

        return out



class Fcn_Vgg16(nn.Module):
    def __init__(self, module_type='32s', n_classes=1, pretrained=True):
        super(Fcn_Vgg16, self).__init__()
        self.n_classes = n_classes
        self.module_type = module_type

        # VGG16=2+2+3+3+3+3
        # VGG16网络的第一个模块是两个out_channel=64的卷积块
        self.conv1_block = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),        #输入3通道，输出64通道，卷积核大小为3，用100填充
            nn.ReLU(inplace=True),                   #inplace=True，节省内存
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True), #核的大小为2，步长为2，向上取整
        )

        # VGG16网络的第二个模块是两个out_channel=128的卷积块
        self.conv2_block = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        # VGG16网络的第三个模块是三个out_channel=256的卷积块
        self.conv3_block = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        # VGG16网络的第四个模块是三个out_channel=512的卷积块
        self.conv4_block = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        # VGG16网络的第五个模块是三个out_channel=512的卷积块
        self.conv5_block = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.n_classes, 1),
        )

        if self.module_type=='16s' or self.module_type=='8s':
            self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        if self.module_type=='8s':
            self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)

        if pretrained:
            self.init_vgg16()

    def init_vgg16(self):
        vgg16 = models.vgg16(pretrained=True)           #获得已经训练好的模型

        # -----------赋值前面2+2+3+3+3层feature的特征-------------
        # 由于vgg16的特征是Sequential，获得其中的子类通过children()
        vgg16_features = list(vgg16.features.children())
        #print(vgg16_features)
        conv_blocks = [self.conv1_block, self.conv2_block, self.conv3_block, self.conv4_block, self.conv5_block]
        conv_ids_vgg = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 30]]  #对应VGG的五个块

        for conv_block_id, conv_block in enumerate(conv_blocks):
            #print(conv_block_id)
            conv_id_vgg = conv_ids_vgg[conv_block_id]
            #print(conv_id_vgg)
            # zip函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象，建立了FCN网络与VGG网络的对应关系。
            for l1, l2 in zip(conv_block, vgg16_features[conv_id_vgg[0]:conv_id_vgg[1]]):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    # 将网络对应的权重由训练好的VGG赋值给FCN
                    l1.weight.data = l2.weight.data
                    l1.bias.data = l2.bias.data
                    # print(l1)
                    # print(l2)

        # -----------赋值后面3层classifier的特征-------------
        vgg16_classifier = list(vgg16.classifier.children())
        for l1, l2 in zip(self.classifier, vgg16_classifier[0:3]):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Linear):
                l1.weight.data = l2.weight.data.view(l1.weight.size())
                l1.bias.data = l2.bias.data.view(l1.bias.size())

        # -----赋值后面1层classifier的特征，由于类别不同，需要修改------
        l1 = self.classifier[6]
        l2 = vgg16_classifier[6]
        if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Linear):
            l1.weight.data = l2.weight.data[:self.n_classes, :].view(l1.weight.size())
            l1.bias.data = l2.bias.data[:self.n_classes].view(l1.bias.size())

    def forward(self, x):
        '''

        :param x: (1, 3, 360, 480)
        :return:
        '''
        conv1 = self.conv1_block(x)
        conv2 = self.conv2_block(conv1)
        conv3 = self.conv3_block(conv2)
        conv4 = self.conv4_block(conv3)
        conv5 = self.conv5_block(conv4)
        score = self.classifier(conv5)
        #print('score', score.shape)  #[1, 21, 12, 16]

        if self.module_type=='16s' or self.module_type=='8s':
            score_pool4 = self.score_pool4(conv4)    #[1, 21, 35, 43]
            #print('pool4',score_pool4.shape)
        if self.module_type=='8s':
            score_pool3 = self.score_pool3(conv3)    #[1, 21, 70, 85]
            #print('pool3', score_pool3.shape)
        # print(conv1.data.size())
        # print(conv2.data.size())
        # print(conv4.data.size())
        # print(conv5.data.size())
        # print(score.data.size())
        # print(x.data.size())
        if self.module_type=='16s' or self.module_type=='8s':
            # 双线性插值，由[1, 21, 12, 16]扩大到[1, 21, 35, 43]
            score = F.interpolate(score, score_pool4.size()[2:], mode='bilinear', align_corners=True)
            score += score_pool4
        if self.module_type=='8s':
            # 双线性插值，由[1, 21, 35, 43]扩大到[1, 21, 70, 85]
            score = F.interpolate(score, score_pool3.size()[2:], mode='bilinear', align_corners=True)
            score += score_pool3
        # 双线性插值，由[1, 21, 35, 43]扩大到[1, 21, 360, 480]
        out = F.interpolate(score, x.size()[2:], mode='bilinear', align_corners=True)
        # sigmoid=nn.Sigmoid()
        # out=sigmoid(out)
        return out

class VGG16(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        base_model = models.vgg16(pretrained=False)

        self.layers = list(base_model.children())
        layers=self.layers
        self.layer1 = nn.Sequential(*layers[:1])  # size=(N, 64, x.H/2, x.W/2)
        # self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        # self.layer2 = layers[5]  # size=(N, 128, x.H/4, x.W/4)
        # self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear')
        # self.layer3 = layers[6]  # size=(N, 256, x.H/8, x.W/8)
        # self.upsample3 = nn.Upsample(scale_factor=16, mode='bilinear')
        # self.layer4 = layers[7]  # size=(N, 512, x.H/16, x.W/16)
        # self.upsample4 = nn.Upsample(scale_factor=32, mode='bilinear')
        #
        # self.conv1k = nn.Conv2d(64 + 128 + 256 + 512, n_class, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        up1 = self.upsample1(x)
        x = self.layer2(x)
        up2 = self.upsample2(x)
        x = self.layer3(x)
        up3 = self.upsample3(x)
        x = self.layer4(x)
        up4 = self.upsample4(x)

        merge = torch.cat([up1, up2, up3, up4], dim=1)
        merge = self.conv1k(merge)
        #out = self.sigmoid(merge)
        out=merge

        return out


if __name__ == '__main__':
    net=VGG16(1)
    print(net.layers)
    print('layer1')
    print(net.layer1)
