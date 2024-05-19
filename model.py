import torch
import torch.nn as nn
import torch.nn.functional as F
from dataparser import parser_args
Arg = parser_args()
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
#        x = x.view(-1, 16 * 5 * 5)
        x = x.view(-1, x.shape[0])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

class MLeNet(nn.Module):
    def __init__(self):
        super(MLeNet, self).__init__()
        # 定义卷定义层卷积层,1个输入通道，6个输出通道，5*5的filter,28+2+2=32
        # 左右、上下填充padding
        # MNIST图像大小28，LeNet大小是32
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        # 定义第二层卷积层
        self.conv2 = nn.Conv2d(6, 16, 5)

        # 定义3个全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 前向传播
    def forward(self, img):
        # 先卷积，再调用relue激活函数，然后再最大化池化
        x = F.max_pool2d(F.relu(self.conv1(img)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # num_flat_features=16*5*5
        # x = x.view(-1, self.num_flat_features(x))

        # 第一个全连接
        x = F.relu(self.fc1(x.view(img.shape[0], -1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return nn.functional.softmax(x, dim=1)


class ModelUtils():
    def __init__(self):
        self.path = Arg.save_path + "/models/"

    def save_model(self,Model_name,model):
        Model_path = self.path + Model_name
        torch.save(model.state_dict(), Model_path)
        print("模型保存成功~路径为" + Model_path)

    def load_model(self,Model_name,model):
        Model_path = self.path + Model_name
        model_params = torch.load(Model_path)
        model.load_state_dict(model_params)
        return model
