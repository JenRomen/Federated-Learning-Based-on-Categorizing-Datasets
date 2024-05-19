import torch
import torchvision
from torchvision import datasets, transforms

def downData(name="MNIST"):
    train_dataset = torchvision.datasets.MNIST(root='data', train=True,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor()
                                               ]), download=True)
    test_dataset = torchvision.datasets.MNIST(root='data', train=False, transform=transforms.Compose([
                                                   transforms.ToTensor()
                                               ]))
    return train_dataset, test_dataset


def splitDataByClass(dataset_name='MNIST'):
    train_dataset, test_dataset = downData(dataset_name)

    sorted_indices = torch.argsort(train_dataset.targets)
    sorted_data = train_dataset.data[sorted_indices]
    sorted_targets = train_dataset.targets[sorted_indices]

    data_by_class = {}
    targets_by_class = {}
    for i in range(10):
        indices = (sorted_targets == i).nonzero(as_tuple=True)[0]
        data_by_class[i] = sorted_data[indices].type(torch.float32).view(len(sorted_data[indices]), 1, 28, 28)
        targets_by_class[i] = sorted_targets[indices].type(torch.long).view(len(sorted_targets[indices]))

    return data_by_class, targets_by_class, (
    test_dataset.data.type(torch.float32).view(len(test_dataset.data), 1, 28, 28)
    , test_dataset.targets.type(torch.long).view(len(test_dataset.targets)))

def fedavg_updata_weight(model,client_list,global_change_model):
    n = 1/len(client_list)
    for i in range(len(client_list)):
        client_list[i].get()

    # 计算模型的聚合更新
    for name, param in global_change_model.named_parameters():
        data = 0
        for i in range(len(client_list)):
            client_list[i].state_dict()[name].copy_(client_list[i].state_dict()[name].data - model.state_dict()[name].data)
            data += client_list[i].state_dict()[name].data
            #client_list[i].state_dict()[name].copy_(client_list[i].state_dict()[name].data + model.state_dict()[name].data)#加这句返回的就是模型而不是更新
        with torch.no_grad():
            global_change_model.state_dict()[name].copy_(data * n)
    #修改全局模型参数
    for name, param in model.named_parameters():
        data = model.state_dict()[name].data + global_change_model.state_dict()[name].data
        with torch.no_grad():
            model.state_dict()[name].copy_(data)

    return model,global_change_model,client_list


def distribute_data_class(data_by_class, targets_by_class):
    client_data = []

    # 遍历每个类别
    for i in range(10):
        data = data_by_class[i]
        targets = targets_by_class[i]
        client_data.append((data, targets))

    return client_data
