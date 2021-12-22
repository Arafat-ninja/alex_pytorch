import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):    #数据集类别个数
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(       #sequ用来精简代码
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]  深度3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),    #num_classes数据集类别个数
        )
        if init_weights:           #搭建网络中如果传入初始化权重的话True，就会进入条件
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)   #从第一位展平
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():      #首先遍历modules这个模块
            if isinstance(m, nn.Conv2d):    #判断得到的层结构是否等于给定的层结构是否
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') #对卷积层权重进行初始化
                if m.bias is not None:    #如果偏置不为空
                    nn.init.constant_(m.bias, 0)     #用零进行赋值
            elif isinstance(m, nn.Linear):    #判断是否为全连接层
                nn.init.normal_(m.weight, 0, 0.01)   #正态分布
                nn.init.constant_(m.bias, 0)
