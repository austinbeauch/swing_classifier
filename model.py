import torch
from torch import nn
import torch.nn.functional as F


class BadModel(nn.Module):

    def __init__(self, num_conv_outer, num_conv_inner, ksize, 
                 num_hidden=1, num_unit=100, nchannel_base=6, num_class=10):
        super(BadModel, self).__init__()
        
        # 6 dims from ax,ay,az,gx,gy,gz
        indim = 6
        
        # TODO: Set this to whatever the truncated input length is
        # or better yet, determine it dynamically 
        cur_w = 500
        
        self.num_hidden = num_hidden
        self.num_conv_outer = num_conv_outer
        self.num_conv_inner = num_conv_inner
        
        outdim = None
        for _i in range(num_conv_outer):
            outdim = nchannel_base * 2 ** _i
            for _j in range(num_conv_inner):
                setattr(self, f"conv_conv1d_{_i}_{_j}", nn.Conv1d(indim, outdim, ksize))
                setattr(self, f"conv_relu_{_i}_{_j}", nn.ReLU())
                cur_w = (cur_w - ksize) + 1
                indim = outdim

            setattr(self, f"conv_pool_{_i}", nn.MaxPool1d(2, 2))
            cur_w = (cur_w - 2) // 2 + 1

        indim = outdim * cur_w
        for _i in range(num_hidden):
            outdim = num_unit
            setattr(self, f"fc_linear_{_i}", nn.Linear(indim, outdim))
            setattr(self, f"fc_relu_{_i}", nn.ReLU())
            indim = outdim
            
        self.output = nn.Linear(indim, num_class)
        self.softmax = nn.Softmax(dim=1)

#         print(self)
        
    def forward(self, x):
        for _i in range(self.num_conv_outer):
            for _j in range(self.num_conv_inner):
                x = getattr(self, f"conv_conv1d_{_i}_{_j}")(x)
                x = getattr(self, f"conv_relu_{_i}_{_j}")(x)
            x = getattr(self, f"conv_pool_{_i}")(x)

        # Flatten
        x = x.view(x.shape[0], -1)
        
        # Apply the fully connected layers
        for _i in range(self.num_hidden):
            x = getattr(self, "fc_linear_{}".format(_i))(x)
            x = getattr(self, "fc_relu_{}".format(_i))(x)
        x = self.output(x)

        # softmax on the swing type classes
        swing_type = self.softmax(x[:,:-1])
        
        # concatenate back with the distance regression
        x = torch.cat([swing_type, x[:,-1:]], axis=1)
        
        return x


class BadModel2(nn.Module):

    def __init__(self):
        super(BadModel2, self).__init__()
        # kernel
        self.conv1 = nn.Conv2d(1, 28, (3, 5), stride=(3, 1))
        self.conv2 = nn.Conv2d(28, 56, (2, 5), stride=(1, 1))
        self.conv3 = nn.Conv2d(56, 56, (1, 5), stride=(1, 1))
        
        self.fc1_cls = nn.Linear(56 * 59, 512)  
        self.fc2_cls = nn.Linear(512, 256)
        self.fc3_cls = nn.Linear(256, 128)
        self.fc4_cls = nn.Linear(128, 9)
        # self.softmax = nn.Softmax(dim=1)
        
        self.fc1_dist = nn.Linear(56 * 59, 256)
        self.fc2_dist = nn.Linear(256, 128)
        self.fc3_dist = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(0)

    def forward(self, x):
        
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (1,2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (1,2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, (1,2))
                        
        x = x.view(-1, self.num_flat_features(x))
        
        x_cls = F.relu(self.fc1_cls(x))
        x_cls = self.dropout(x_cls)
        x_cls = F.relu(self.fc2_cls(x_cls))
        x_cls = self.dropout(x_cls)
        x_cls = F.relu(self.fc3_cls(x_cls))
        x_cls = self.dropout(x_cls)
        x_cls = self.fc4_cls(x_cls)
        # x_cls = self.softmax(x_cls)

        x_dist = F.relu(self.fc1_dist(x))
        x_cls = self.dropout(x_dist)
        x_dist = F.relu(self.fc2_dist(x_dist))
        x_cls = self.dropout(x_dist)
        x_dist = self.fc3_dist(x_dist)

        x = torch.cat([x_cls, x_dist], axis=1)
        
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features