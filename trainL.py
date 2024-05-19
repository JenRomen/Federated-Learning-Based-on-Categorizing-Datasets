import syft as sy
import torch
from dataparser import parser_args
import torch.nn.functional as F

hook = sy.TorchHook(torch)
args = parser_args()

def test(model, test_data, device):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        test_dataset = torch.utils.data.TensorDataset(test_data[0],test_data[1])
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
        for epoch_ind, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            # test_loss += loss_function(output, target).item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_data[0])
        print('\nTest set : Average loss : {:.4f}, Accuracy: {}/{} ( {:.2f}%)\n'.format(
            test_loss, correct, len(test_data[0]),
            100. * correct / len(test_data[0])))
    return test_loss,100. * correct / len(test_data[0])

def distribute_data(client_number,model):
    client_model = []
    client_optim = []
    for i in range(client_number):
        client_model.append(model.copy())
        client_optim.append(torch.optim.SGD(client_model[i].parameters(), lr=args.lr, momentum=args.momentum))
    return client_model,client_optim,model.copy()

def fedavg_updata_weight(model,client_list,global_change_model):
    n = 1/len(client_list)
    for i in range(len(client_list)):
        client_list[i].get()

    # 计算模型的聚合更新
    for name, param in global_change_model.named_parameters(): # 原model中每个部分的权重，如conv1.weight tensor([...])
        data = 0
        for i in range(len(client_list)):
            client_list[i].state_dict()[name].copy_(client_list[i].state_dict()[name].data - model.state_dict()[name].data)
            data += client_list[i].state_dict()[name].data
        with torch.no_grad():
            global_change_model.state_dict()[name].copy_(data * n) # 累加所有的客户端参数-原model参数，即gk，取平均
    #修改全局模型参数
    for name, param in model.named_parameters():
        data = model.state_dict()[name].data + global_change_model.state_dict()[name].data
        with torch.no_grad():
            model.state_dict()[name].copy_(data)

    return model,global_change_model,client_list

def federated_train(client_number,model,client_data,test_data,Var, device):
    print("----------3.1 创建数量为{}的客户端----------".format(client_number))
    client_list = []
    for i in range(client_number):
        client_list.append(sy.VirtualWorker(hook, id=str(i)))
    print("----------创建成功----------")
    torch.manual_seed(args.seed)
    print("----------3.2 分配模型----------")
    client_model, client_optim, global_change_model = distribute_data(client_number, model)
    print("----------分配成功----------")
    print("----------3.3 开始训练----------")
    for i in range(client_number):
        client_model[i].train()
        client_model[i].send(client_list[i])

    for i in range(client_number):
        print("----------第{}个客户端开启训练----------".format(i + 1))
        train_dataset = torch.utils.data.TensorDataset(client_data[i][0], client_data[i][1])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        mean_loss = 0
        for epoch in range(args.local_epochs):#客户端训练次数 3次
            summ = 0
            k = 0
            for epoch_ind, (data, target) in enumerate(train_loader):
                k += 1
                data = data.send(client_list[i]).to(device)
                target = target.send(client_list[i]).to(device)
                client_optim[i].zero_grad()
                pred = client_model[i](data)
                loss = F.cross_entropy(pred, target)

                loss.backward()
                client_optim[i].step()
                value = loss.get().data.item()
                summ += value
                #if k % 100 == 0:
                    #print("第{}个客户端第{}个epoch的loss: {}")
                if epoch_ind % 30 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, epoch_ind * args.batch_size, len(train_loader) * args.batch_size,
                                   100. * epoch_ind / len(train_loader), value))
            mean_loss = summ/len(train_dataset)
            #print(f"第{i+1}个客户端第{epoch+1}个epoch的loss: {mean_loss}")
            if epoch == 0:
                Var.insert_var("user_loss_start_epoch",mean_loss, format="list", epoch=True)
            if epoch == args.local_epochs - 1:
                Var.insert_var("user_loss_end_epoch",mean_loss, format="list", epoch=True)
        Var.insert_var("train_loss_epoch",mean_loss,format = "list",epoch= True)
    # 10个客户端本地训练完成后
    with torch.no_grad():
        # 更新权重
        model,global_change_model,client_model = fedavg_updata_weight(model, client_model,global_change_model)
        loss,acc = test(model, test_data, device)
        Var.insert_var("client_model", client_model, format="list")
        Var.insert_var("acc", acc, "list")
        Var.insert_var("global_change_model", global_change_model,"list")
    return model




