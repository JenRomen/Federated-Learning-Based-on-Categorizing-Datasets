import torch.nn.functional as F

def train(args, model, device, dataloader, optimizer, epoch_num):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        model.send(data.location)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        model.get()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_num, batch_idx * args.batch_size, len(dataloader) * args.batch_size,
                           100. * batch_idx / len(dataloader), loss.get()))



