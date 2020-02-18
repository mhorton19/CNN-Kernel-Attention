
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _pair, _single 
from torch.nn import init
import torchvision
import torchvision.transforms as transforms
import math
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

NCLASSES = 10

#implementation from https://github.com/mf1024/Batch-Renormalization-PyTorch/blob/master/batch_renormalization.py
class BatchRenormalization2D(nn.Module):

    def __init__(self, num_features,  eps=1e-05, momentum=0.01, r_d_max_inc_step = 0.0001):
        super(BatchRenormalization2D, self).__init__()

        self.eps = eps
        self.momentum = torch.tensor( (momentum), requires_grad = False)

        self.gamma = torch.nn.Parameter(torch.ones((1, num_features, 1, 1)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1)), requires_grad=True)

        self.running_avg_mean = torch.ones((1, num_features, 1, 1), requires_grad=False)
        self.running_avg_std = torch.zeros((1, num_features, 1, 1), requires_grad=False) 

        self.max_r_max = 3.0
        self.max_d_max = 5.0

        self.r_max_inc_step = r_d_max_inc_step
        self.d_max_inc_step = r_d_max_inc_step

        self.r_max = torch.tensor( (1.0), requires_grad = False)
        self.d_max = torch.tensor( (0.0), requires_grad = False)

    def forward(self, x):

        device = self.gamma.device

        batch_ch_mean = torch.mean(x, dim=(0,2,3), keepdim=True).to(device)
        batch_ch_std = torch.clamp(torch.std(x, dim=(0,2,3), keepdim=True), self.eps, 1e10).to(device)

        self.running_avg_std = self.running_avg_std.to(device)
        self.running_avg_mean = self.running_avg_mean.to(device)
        self.momentum = self.momentum.to(device)

        self.r_max = self.r_max.to(device)
        self.d_max = self.d_max.to(device)


        if self.training:

            r = torch.clamp(batch_ch_std / self.running_avg_std, 1.0 / self.r_max, self.r_max).to(device).data.to(device)
            d = torch.clamp((batch_ch_mean - self.running_avg_mean) / self.running_avg_std, -self.d_max, self.d_max).to(device).data.to(device)

            x = ((x - batch_ch_mean) * r )/ batch_ch_std + d
            x = self.gamma * x + self.beta

            if self.r_max < self.max_r_max:
                self.r_max += self.r_max_inc_step * x.shape[0]

            if self.d_max < self.max_d_max:
                self.d_max += self.d_max_inc_step * x.shape[0]

        else:

            x = (x - self.running_avg_mean) / self.running_avg_std
            x = self.gamma * x + self.beta

        self.running_avg_mean = self.running_avg_mean + self.momentum * (batch_ch_mean.data.to(device) - self.running_avg_mean)
        self.running_avg_std = self.running_avg_std + self.momentum * (batch_ch_std.data.to(device) - self.running_avg_std)

        return x

class _AttentionConvNd(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                 'padding_mode', 'output_padding', 'in_channels',
                 'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels, kernel_size, num_kernels, vector_length, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super(_AttentionConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.vector_length = vector_length
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode

        if transposed:
            self.weight_shape = (in_channels, out_channels // groups, *kernel_size)
        else:
            self.weight_shape = (out_channels, in_channels // groups, *kernel_size)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.kernel_set = Parameter(torch.Tensor(num_kernels, *kernel_size))
        self.kernel_vectors = Parameter(torch.Tensor(vector_length, num_kernels))

        self.input_vectors = Parameter(torch.Tensor(self.weight_shape[1], vector_length))
        self.output_vectors = Parameter(torch.Tensor(self.weight_shape[0], vector_length))
        
        self.fc_attention = nn.Linear(in_channels, vector_length)
        self.dropout = nn.Dropout(p=0.1)

        self.attention_factor = 1/math.sqrt(vector_length)

        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.kernel_set, 6/math.sqrt(self.weight_shape[0]))

        init.normal_(self.input_vectors)
        init.normal_(self.output_vectors)
        init.normal_(self.kernel_vectors)

        if self.bias is not None:
            fan_in = self.weight_shape[0]
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_AttentionConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


    def generate_kernels(self, input):
        global epoch_end

        pooled_inputs = self.dropout(nn.AvgPool2d(input.size()[2:3])(input)[:,:,0,0])
        gap_attention_vector = self.fc_attention(pooled_inputs).unsqueeze(1).unsqueeze(1)

        input_vectors_exp = self.input_vectors.unsqueeze(0)
        output_vectors_exp = self.output_vectors.unsqueeze(1)

        pairwise_attention_vectors = (input_vectors_exp*output_vectors_exp).unsqueeze(0)

        batch_attention_vectors = (pairwise_attention_vectors*gap_attention_vector)

        batch_attention_vectors_flattened = batch_attention_vectors.view(-1, self.vector_length)

    
        attention_scalars = F.softmax(self.attention_factor*torch.matmul(batch_attention_vectors_flattened, self.kernel_vectors), dim = -1)


        kernels = torch.matmul(attention_scalars, self.kernel_set.view(self.num_kernels, -1)).view(input.size()[0], *self.weight_shape)
    
        
        return kernels
    


class AttentionConv2d(_AttentionConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, num_kernels=100, vector_length=20, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(AttentionConv2d, self).__init__(
            in_channels, out_channels, kernel_size, num_kernels, vector_length, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def _conv_forward(self, input, weight, total_groups):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, total_groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, total_groups)


    #In order to efficiently perform a convolutoin with a different kernel set for each input in the batch, this appends all of the kernels and append all of the inputs along the channel dimension
    #Then, this exploits group convolutions to pair each input with its corresponding kernel set efficiently
    def forward(self, input):
        batch_size = input.size()[0]
        input_channels = input.size()[1]

        kernels = self.generate_kernels(input)

        kernels = kernels.view(kernels.size()[0]*kernels.size()[1], kernels.size()[2], kernels.size()[3], kernels.size()[4])

        transformed_input = input.view(1, batch_size*input_channels, input.size()[2], input.size()[3])
        
        output = self._conv_forward(transformed_input, kernels, self.groups*batch_size)
        return output.view(batch_size, self.out_channels, output.size(2), output.size(3))

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
    bs = 8
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

 
    

    
    
    class KernelGenerateCNN(nn.Module):
        def __init__(self):
            super(KernelGenerateCNN, self).__init__()
            
            num_channels = 100
            num_kernels = 50
            vector_length = 100

            groups = 1
            self.num_channels = num_channels
            self.num_kernels = num_kernels
            self.conv1_1 = nn.Conv2d(3, 20, 3, padding=1, bias=False)
            self.bn1_1 = BatchRenormalization2D(20)
            self.cond_conv1_1 = AttentionConv2d(20, 50, 3, vector_length=vector_length, num_kernels=num_kernels, padding=1, groups=groups, bias=False)
            self.cond_bn1_1 = BatchRenormalization2D(50)
            self.conv1_2 = AttentionConv2d(50, 70, 2, vector_length=vector_length, num_kernels=num_kernels, stride=2, bias=False)
            self.bn1_2 = BatchRenormalization2D(70)
            
            self.cond_conv2_1 = AttentionConv2d(70, 90, 3, vector_length=vector_length, num_kernels=num_kernels, padding=1, groups=groups, bias=False)
            self.cond_bn2_1 = BatchRenormalization2D(90)
            self.conv2_1 = AttentionConv2d(90, 100, 2, vector_length=vector_length, num_kernels=num_kernels, stride=2, bias=False)
            self.bn2_1 = BatchRenormalization2D(100)

            self.cond_conv3_1 = AttentionConv2d(100, 100, 3, vector_length=vector_length, num_kernels=num_kernels, padding=1, groups=groups, bias=False)
            self.cond_bn3_1 = BatchRenormalization2D(100)
            
            self.cond_conv3_2 = AttentionConv2d(100, 100, 3, vector_length=vector_length, num_kernels=num_kernels, padding=1, groups=groups, bias=False)
            self.cond_bn3_2 = BatchRenormalization2D(100)
            
            self.cond_conv3_3 = AttentionConv2d(100, 100, 3, vector_length=vector_length, num_kernels=num_kernels, padding=1, groups=groups, bias=False)
            self.cond_bn3_3 = BatchRenormalization2D(100)
            
            self.global_pool = nn.AvgPool2d(8, 8)
            self.fc = nn.Linear(num_channels, NCLASSES)
            
            
        def forward(self, x):   
            #print(x)
            global second_layer
            x = F.relu(self.bn1_1(self.conv1_1(x)))
            x = F.relu(self.cond_bn1_1(self.cond_conv1_1(x)))
            x = F.relu(self.bn1_2(self.conv1_2(x)))

            x = F.relu(self.cond_bn2_1(self.cond_conv2_1(x)))
            x = F.relu(self.bn2_1(self.conv2_1(x)))
            
            x = F.relu(self.cond_bn3_1(self.cond_conv3_1(x)))
            x = F.relu(self.cond_bn3_2(self.cond_conv3_2(x)))
            second_layer = True
            x = F.relu(self.cond_bn3_3(self.cond_conv3_3(x)))
            
            x = self.global_pool(x).view(-1, self.num_channels)
            x = self.fc(x)
            return x


    net = KernelGenerateCNN()


    net = net.to(device)


    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    MIN_LR = 0.0
    MAX_LR = 0.01
    steps_per_epoch = len(trainloader)
    steps_up = 100000
    steps_down = steps_up


    schedule = torch.optim.lr_scheduler.CyclicLR(optimizer, MIN_LR, MAX_LR, step_size_up=steps_up, step_size_down=steps_down, cycle_momentum=False, base_momentum=0.0, max_momentum=0.0)


    print_increment = 600
    val_increment = 1000
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

            schedule.step()


            # print statistics
            running_loss += loss.item()

            outputs_np = np.argmax(outputs.data.cpu().numpy(), 1)
            correct = np.mean(outputs_np == labels.data.cpu().numpy())
            running_accuracy += correct
            if i % print_increment == print_increment-1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.5f accuracy %.5f' %
                      (epoch + 1, i + 1, running_loss / print_increment, running_accuracy/print_increment))
                running_loss = 0.0
                running_accuracy = 0.0
                print('LEARNING RATE: ' + str(schedule.get_lr()[0]))

             

        testdataiter = iter(testloader)
        (loss, acc) = validation(net, testdataiter, criterion)
        
        print('EPOCH %d VAL loss: %.5f accuracy %.5f' %
                      (epoch + 1, loss, acc))


        

    print('Finished Training')