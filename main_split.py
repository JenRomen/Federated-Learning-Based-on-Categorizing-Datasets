import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from argparse import ArgumentParser
import syft as sy
from dataparser import parser_args
from model import MLeNet as Net
from model import ModelUtils
from train import train
from test import test
from splitMINIST import splitDataByClass, distribute_data_class
from trainL import federated_train, fedavg_updata_weight, test
from writelog import Global_Variable, DataUtils


dataUtils = DataUtils()
args = parser_args()
modelUtils = ModelUtils()
Var = Global_Variable(args.client_num)
def save_data(Var):
    dataUtils.insert_data_to_excel("train_loss_epoch.xlsx", Var.get_var("train_loss_epoch")[-1], sheet_name="Sheet")
    dataUtils.insert_data_to_excel("acc.xlsx", [Var.get_var("acc")[-1]], sheet_name="Sheet")
    dataUtils.insert_data_to_excel("user_loss_start_epoch.xlsx", Var.get_var("user_loss_start_epoch")[-1], sheet_name="Sheet")
    dataUtils.insert_data_to_excel("user_loss_end_epoch.xlsx",Var.get_var("user_loss_end_epoch")[-1], sheet_name="Sheet")


if __name__ == "__main__":

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    model = Net().to(device)
    print("----------1.处理数据阶段----------")
    data_by_class, targets_by_class, test_dataset = splitDataByClass("MINIST")
    print("数据处理完成")
    print("----------2.分发数据阶段----------")
    client_number = args.client_num
    client_data = distribute_data_class(data_by_class, targets_by_class)
    print("数据分发完毕")
    print("----------3.联邦学习阶段----------")
    for i in range(args.epochs):
        print(f"----------epoch={i + 1}----------")
        model = federated_train(client_number, model, client_data, test_dataset, Var, device)
        save_data(Var)
        if i % 10 == 0:
            print("每10回合的模型保存")
            modelUtils.save_model("fedAvg.pt", model)
    print("----------4.测试阶段----------")
    test(model, test_dataset, device)
    print("----------5.模型保存----------")
    modelUtils.save_model("fedAvg.pt", model)
    dataUtils.save_var("var_fedavg", Var)







