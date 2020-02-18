
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd, _pair, _single 
import torchvision
import torchvision.transforms as transforms
import math
import random
import numpy as np

VECTOR_SIZE = 9
NCLASSES = 10


def validation(model, testloader, criterion):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        accuracy = 0
        iterations = 0
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model.forward(images)
            test_loss += criterion(outputs, labels).item()

            outputs_np = np.argmax(outputs.data.cpu().numpy(), 1)
            #print(outputs_np)
            correct = np.mean(outputs_np == labels.data.cpu().numpy())
            accuracy += correct
            iterations += 1

        model.train()
        return test_loss/iterations, accuracy/iterations

if __name__ == '__main__':
    device = torch.device("cuda:0")
    bs = 32
    #torch.multiprocessing.freeze_support()
    aug_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.RandomCrop(size=[32,32], padding=4),
         transforms.RandomAffine(180, scale=(0.8, 1.2), shear=(20, 20)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='cifar-10-python', train=True,
                                            download=True, transform=aug_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='cifar-10-python', train=False,
                                           download=True, transform=aug_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs,
                                             shuffle=False, num_workers=0)


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()


    import torch.nn as nn
    import torch.nn.functional as F

    class KernelGenerator(nn.Module):
        def __init__(self, input_params, vector_size, channels_used=18):
            super(KernelGenerator, self).__init__()
            self.channels_used = channels_used

            self.fc_out = nn.Linear(channels_used, vector_size)
        
        def forward(self, params, attention_weights, channels):
            used_channels = channels[:,:self.channels_used,0,0]
            attention_vector = self.fc_out(used_channels)

            attention_vector = attention_vector.unsqueeze(1).unsqueeze(-1)

            attention_scalar = torch.sigmoid(torch.matmul(attention_weights, attention_vector).unsqueeze(-1))
            
            
            return params*attention_scalar
    
    class ConditionalConv2d(_ConvNd):
        def __init__(self, in_channels, out_channels, kernel_size, kernel_parameters=9, vector_size=9, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, channels_used=128, padding_mode='zeros'):
            kernel_size = _pair(kernel_size)
            stride = _pair(stride)
            padding = _pair(padding)
            dilation = _pair(dilation)
            super(ConditionalConv2d, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation,
                False, _pair(0), groups, bias, padding_mode)

            self.kernel_generator = KernelGenerator(kernel_parameters, vector_size, channels_used=channels_used)
            self.attention_weights = Parameter(torch.Tensor(out_channels, in_channels // groups, vector_size))
            self.reset_attention_parameters()

        def reset_attention_parameters(self):
            nn.init.zeros_(self.attention_weights) 

        def _conv_forward(self, input, weight, total_groups):
            if self.padding_mode == 'circular':
                expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                    (self.padding[0] + 1) // 2, self.padding[0] // 2)
                return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                                weight, self.bias, self.stride,
                                _pair(0), self.dilation, total_groups)
                
            return F.conv2d(input, weight, self.bias, self.stride,
                            self.padding, self.dilation, total_groups)
        
        def generate_kernels(self, input):
            pooled_inputs = nn.AvgPool2d(input.size()[2:3])(input)
            
            
            weights = self.weight.unsqueeze(0)
            weights = weights.expand(input.size(0), -1, -1, -1, -1)

            attention_weights = self.attention_weights.unsqueeze(0)
            attention_weights = attention_weights.expand(input.size(0), -1, -1, -1)
            
            kernel_pred = self.kernel_generator(weights, attention_weights, pooled_inputs)
            
            return kernel_pred

        def forward(self, input):
            batch_size = input.size()[0]
            input_channels = input.size()[1]

            kernels = self.generate_kernels(input)

            kernels = kernels.view(kernels.size()[0]*kernels.size()[1], kernels.size()[2], kernels.size()[3], kernels.size()[4])

            transformed_input = input.view(1, batch_size*input_channels, input.size()[2], input.size()[3])
            
            output = self._conv_forward(transformed_input, kernels, self.groups*batch_size)
            return output.view(batch_size, self.out_channels, output.size(2), output.size(3))

        
    
    
    class KernelGenerateCNN(nn.Module):
        def __init__(self):
            super(KernelGenerateCNN, self).__init__()
            
            num_kernels = 128
            #groups = num_kernels//16
            groups = 1
            self.num_kernels = num_kernels
            self.conv1_1 = nn.Conv2d(3, num_kernels, 3, padding=1, bias=False)
            self.bn1_1 = nn.BatchNorm2d(num_kernels)
            self.cond_conv1_1 = nn.Conv2d(num_kernels, num_kernels, 3, padding=1, groups=groups, bias=False)
            self.cond_bn1_1 = nn.BatchNorm2d(num_kernels)
            self.conv1_2 = nn.Conv2d(num_kernels, num_kernels, 2, stride=2, bias=False)
            self.bn1_2 = nn.BatchNorm2d(num_kernels)
            
            self.cond_conv2_1 = nn.Conv2d(num_kernels, num_kernels, 3, padding=1, groups=groups, bias=False)
            self.cond_bn2_1 = nn.BatchNorm2d(num_kernels)
            self.conv2_1 = nn.Conv2d(num_kernels, num_kernels, 2, stride=2)
            self.bn2_1 = nn.BatchNorm2d(num_kernels)

            self.cond_conv3_1 = ConditionalConv2d(num_kernels, num_kernels, 3, padding=1, groups=groups, bias=False)
            self.cond_bn3_1 = nn.BatchNorm2d(num_kernels)
            
            self.cond_conv3_2 = ConditionalConv2d(num_kernels, num_kernels, 3, padding=1, groups=groups, bias=False)
            self.cond_bn3_2 = nn.BatchNorm2d(num_kernels)
            
            self.cond_conv3_3 = ConditionalConv2d(num_kernels, num_kernels, 3, padding=1, groups=groups, bias=False)
            self.cond_bn3_3 = nn.BatchNorm2d(num_kernels)
            
            self.global_pool = nn.AvgPool2d(8, 8)
            self.fc = nn.Linear(num_kernels, NCLASSES)
            
        def forward(self, x):   
            x = F.relu(self.bn1_1(self.conv1_1(x)))
            x = F.relu(self.cond_bn1_1(self.cond_conv1_1(x)))
            x = F.relu(self.bn1_2(self.conv1_2(x)))

            x = F.relu(self.cond_bn2_1(self.cond_conv2_1(x)))
            x = F.relu(self.bn2_1(self.conv2_1(x)))
            
            x = F.relu(self.cond_bn3_1(self.cond_conv3_1(x)))
            x = F.relu(self.cond_bn3_2(self.cond_conv3_2(x)))
            x = F.relu(self.cond_bn3_3(self.cond_conv3_3(x)))
            x = self.global_pool(x).view(-1, self.num_kernels)
            x = self.fc(x)
            return x
  


    net = KernelGenerateCNN()

    net = net.to(device)

    
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), amsgrad=True)

    print_increment = 50
    for epoch in range(300):  # loop over the dataset multiple times
        running_loss = 0.0
        running_accuracy = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            outputs_np = np.argmax(outputs.data.cpu().numpy(), 1)

            correct = np.mean(outputs_np == labels.data.cpu().numpy())
            running_accuracy += correct
            if i % print_increment == print_increment-1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.5f accuracy %.5f' %
                      (epoch + 1, i + 1, running_loss / print_increment, running_accuracy/print_increment))
                running_loss = 0.0
                running_accuracy = 0.0

             

        testdataiter = iter(testloader)
        (loss, acc) = validation(net, testdataiter, criterion)
        
        print('EPOCH %d VAL loss: %.5f accuracy %.5f' %
                      (epoch + 1, loss, acc))



        

    print('Finished Training')
