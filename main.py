import torch.optim as optim
import torch
import sys
import torch.nn as nn
# from model_nod3 import Net
from loader import test_loader
from loader import train_loader
import model
from logger import log

# writer = SummaryWriter('./test/')
cuda_avail = torch.cuda.is_available()
# model = Net()

device = "cuda:0" if cuda_avail else "cpu"

m = nn.Sigmoid().to(device)

# optimizer = optim.SGD(model.parameters(), lr= 0.001, momentum= 0.09)
loss_bce = nn.BCEWithLogitsLoss().to(device)
loss_l1 = nn.SmoothL1Loss().to(device)

def channel_max(tensor):
    max_i = torch.zeros(tensor.size(0))
    for i in range(0, tensor.size(0)):
        max_i[i] = torch.argmax(tensor[i, :, :])
    return max_i

def adjust_learning_rate(epoch):
    lr = optimizer.param_groups[0]['lr']
    print(lr)
    if (epoch % 30) == 0:
        lr = lr / 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_models(epoch):
    torch.save(model.state_dict(), 'model_save/resnet34_{}.model'.format(epoch))
    print('checkpoint saves')

def test(epoch):
    model.eval()
    test_acc = 0.0
    sample_sum = 0.0
    batch_sum = 0.0
    for i, (images, labels) in enumerate(test_loader):
        # print('test here')
        images = images.to(device)
        labels = labels.to(device)
        outputs_cls, _ = model(images)
        
        prediction = m(outputs_cls).ge(0.5).long()
        test_acc += torch.sum(prediction == labels)
        sample_sum += len(labels)
        batch_sum += 1
    test_acc = test_acc.float() / sample_sum

    return test_acc 


def train(num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        batch_sum = 0.0
        sample_sum = 0.0
        train_acc = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # print(i)
            images = images.to(device)
            labels = labels.to(device)
            # print(labels.cpu().data)
            # print(images.cpu().data)
            # print(images.shape)

            optimizer.zero_grad()
            outputs_cls, _ = model(images)
            loss = loss_bce(outputs_cls, labels.float())
            loss.backward()
            optimizer.step()

            prediction = m(outputs_cls).ge(0.5).long()
            train_acc += torch.sum(prediction == labels)
            batch_sum += 1
            sample_sum += len(labels)
            train_loss += loss.data
        train_loss = train_loss / batch_sum
        train_acc = train_acc.float() / sample_sum
        print(sample_sum)

        test_acc = test(epoch)

        print(lamda, 'Epoch {}, Train Acc: {}, TrainLoss: {}, Test Acc: {}'.format(epoch, train_acc, train_loss, test_acc))
        log.info('lambda: {} , TrainAcc: {} , TrainLoss: {} , TestAcc: {}'.format(lamda, train_acc, train_loss, test_acc))
    
    return test_acc


if __name__ == '__main__':
    print('hello1')
    lamda_log = list()
    for lamda in range(50, 10100, 20):
        model = model.resnet34()
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay= 0.0001)
        total_test_acc = train(30)
        lamda_log.append([lamda, total_test_acc])
    with open('log.txt', 'w') as out:
        out.write('\n'.join(lamda_log))