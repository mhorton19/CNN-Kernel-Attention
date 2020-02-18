
import torch
from torch.nn.modules.conv import _ConvNd, _pair, _single 
import torchvision
import torchvision.transforms as transforms
import math
import random
import numpy as np

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
        def __init__(self, input_params, kernel_size, channels_used=9, layer1_size=100):
            super(KernelGenerator, self).__init__()
            self.channels_used = channels_used
            self.kernel_size = kernel_size
            self.flat_kernel_size = kernel_size[0] * kernel_size[1]
            self.fc1 = nn.Linear(input_params, layer1_size)
            #self.bn1 = nn.BatchNorm1d(layer1_size)
            self.fc_gate = nn.Linear(input_params+channels_used, layer1_size)
            self.fc_out = nn.Linear(layer1_size, self.flat_kernel_size)
        
        def forward(self, params, channels):
            used_channels = channels[:,:,:,:self.channels_used]
            combined_input = torch.cat((params, used_channels), -1)

            orig_shape = combined_input.size()
            combined_input = combined_input.view(orig_shape[0]*orig_shape[1]*orig_shape[2], orig_shape[3])
            
            param_shape = params.size()
            flattened_params = params.contiguous().view(param_shape[0]*param_shape[1]*param_shape[2], param_shape[3])

            x = self.fc1(flattened_params)
            x_gate = F.sigmoid(self.fc_gate(combined_input))
            x = x*x_gate
            x = self.fc_out(x)
            return x.view(orig_shape[0], orig_shape[1], orig_shape[2], self.kernel_size[0], self.kernel_size[1])
        
    class ConditionalConv2d(_ConvNd):
        def __init__(self, in_channels, out_channels, kernel_size, kernel_parameters=9, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, channels_used=18, padding_mode='zeros'):
            kernel_size = _pair(kernel_size)
            self.transformed_kernel_size = kernel_size
            kernel_parameters_tuple = _single(kernel_parameters)
            stride = _pair(stride)
            padding = _pair(padding)
            dilation = _pair(dilation)
            
            self.kernel_size = kernel_size
            super(ConditionalConv2d, self).__init__(
                in_channels, out_channels, kernel_parameters_tuple, stride, padding, dilation,
                False, _pair(0), groups, bias, padding_mode)
            
            self.kernel_generator = KernelGenerator(kernel_parameters, kernel_size, channels_used=channels_used)
            
        def reset_parameters(self):
            nn.init.normal_(self.weight)
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)        
        
        
        def generate_kernels(self, input):
            pooled_inputs = nn.AvgPool2d(input.size()[2:3])(input)
            pooled_inputs = pooled_inputs.view(pooled_inputs.size()[0], 1, 1, pooled_inputs.size()[1])
            pooled_inputs = pooled_inputs.expand(pooled_inputs.size()[0], self.weight.size()[0], self.weight.size()[1], pooled_inputs.size()[-1])
            
            weights = self.weight.view(1, self.weight.size()[0], self.weight.size()[1], self.weight.size()[2])
            weights = weights.expand(input.size(0), weights.size()[1], weights.size()[2], weights.size()[3])
            
            kernel_pred = self.kernel_generator(weights, pooled_inputs)
            
            return kernel_pred
        
        def conv2d_forward(self, input, weight):
            if self.padding_mode == 'circular':
                expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                    (self.padding[0] + 1) // 2, self.padding[0] // 2)
                return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                                weight, self.bias, self.stride,
                                _pair(0), self.dilation, self.groups)
            return F.conv2d(input, weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        def forward(self, input):
            kernels = self.generate_kernels(input)
            
            collected_outputs = [];
            for i in range(kernels.size()[0]):
                kernel_set = kernels[i]
                input_curr = input[i:i+1]
                collected_outputs.append(self.conv2d_forward(input_curr, kernel_set))
            
            return torch.cat(collected_outputs, 0)
    
    class KernelGeneratorCNN(nn.Module):
        def __init__(self):
            super(KernelGeneratorCNN, self).__init__()
            
            num_kernels = 128
            groups = num_kernels//8
            self.num_kernels = num_kernels
            self.conv1_1 = nn.Conv2d(3, num_kernels, 3, padding=1)
            self.bn1_1 = nn.BatchNorm2d(num_kernels)
            self.cond_conv1_1 = ConditionalConv2d(num_kernels, num_kernels, 3, padding=1, groups=groups)
            self.cond_bn1_1 = nn.BatchNorm2d(num_kernels)
            self.conv1_2 = nn.Conv2d(num_kernels, num_kernels, 1)
            self.bn1_2 = nn.BatchNorm2d(num_kernels)
        
            self.cond_conv2_1 = ConditionalConv2d(num_kernels, num_kernels, 3, padding=1, groups=groups)
            self.cond_bn2_1 = nn.BatchNorm2d(num_kernels)
            self.conv2_1 = nn.Conv2d(num_kernels, num_kernels, 1)
            self.bn2_1 = nn.BatchNorm2d(num_kernels)

            self.cond_conv3_1 = ConditionalConv2d(num_kernels, num_kernels, 3, padding=1, groups=groups)
            self.cond_bn3_1 = nn.BatchNorm2d(num_kernels)
            self.conv3_1 = nn.Conv2d(num_kernels, num_kernels, 1)
            self.bn3_1 = nn.BatchNorm2d(num_kernels)
            
            self.cond_conv3_2 = ConditionalConv2d(num_kernels, num_kernels, 3, padding=1, groups=groups)
            self.cond_bn3_2 = nn.BatchNorm2d(num_kernels)
            self.conv3_2 = nn.Conv2d(num_kernels, num_kernels, 1)
            self.bn3_2 = nn.BatchNorm2d(num_kernels)
            
            self.cond_conv3_3 = ConditionalConv2d(num_kernels, num_kernels, 3, padding=1, groups=groups)
            self.cond_bn3_3 = nn.BatchNorm2d(num_kernels)
            self.conv3_3 = nn.Conv2d(num_kernels, num_kernels, 1)
            self.bn3_3 = nn.BatchNorm2d(num_kernels)
            
            
            self.pool = nn.MaxPool2d(2, 2)
            self.global_pool = nn.AvgPool2d(8, 8)
            self.fc = nn.Linear(num_kernels, NCLASSES)
            
        def forward(self, x):   
            #print(x)
            x = F.relu(self.bn1_1(self.conv1_1(x)))
            x = F.relu(self.cond_bn1_1(self.cond_conv1_1(x)))
            x = F.relu(self.bn1_2(self.conv1_2(x)))
            x = self.pool(x)
            
            x = F.relu(self.cond_bn2_1(self.cond_conv2_1(x)))
            x = F.relu(self.bn2_1(self.conv2_1(x)))
            x = self.pool(x)
            
            x = F.relu(self.cond_bn3_1(self.cond_conv3_1(x)))
            x = F.relu(self.bn3_1(self.conv3_1(x)))
            x = F.relu(self.cond_bn3_2(self.cond_conv3_2(x)))
            x = F.relu(self.bn3_2(self.conv3_2(x)))
            x = F.relu(self.cond_bn3_3(self.cond_conv3_3(x)))
            x = F.relu(self.bn3_3(self.conv3_3(x)))
            x = self.global_pool(x).view(-1, self.num_kernels)
            x = self.fc(x)
            return x

    net = KernelGeneratorCNN()
    net = net.to(device)
   

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    


    print_increment = 50
    for epoch in range(300):  # loop over the dataset multiple times

        running_loss = 0.0
        running_accuracy = 0.0
        for i, data in enumerate(trainloader, 0):
           
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()


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

   