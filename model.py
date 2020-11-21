import torch
from torch import nn


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
