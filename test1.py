import torch.nn as nn
import torch

import torch
import torch.nn as nn


def build_act_layer(act_type):

    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()

class ElementScale(nn.Module):
 

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

class LFE(nn.Module):
  

    def __init__(self,
                 dim,
                 kernel_size=3,
                 act_type='GELU',
                 ffn_drop=0.):
        super(LFE, self).__init__()

        self.embed_dims = dim
        self.feedforward_channels = int(dim * 4)

        self.fc1 = nn.Conv2d(
            in_channels=dim,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=dim,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = build_act_layer(act_type)

    def feat_decompose(self, x):
       
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
      
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
       

      
        x = self.fc2(x)
        
        x = self.drop(x)
        x = x.permute(0, 3, 2, 1)
        return x



class DSFE(nn.Module):

    def __init__(self, input_size,dim, proj_size=16, num_heads=4, qkv_bias=False, channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkvv = nn.Linear(dim, dim * 4, bias=qkv_bias)

        
        self.E = nn.Linear(input_size, proj_size)
        self.F = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(dim, int(dim // 2))
        self.out_proj2 = nn.Linear(dim, int(dim // 2))

     
        self.caf = LFE(dim=dim)
    
        
        

    def forward(self, x):
        
        x1 = self.caf(x)
       
        x = x.permute(0, 3, 2, 1)
        
        b1,c1,h1,w1 = x.shape
        x = x.reshape(b1,c1,-1).transpose(-1,-2)
        
        B, N, C = x.shape
       

        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)

        qkvv = qkvv.permute(2, 0, 3, 1, 4)

        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_shared_projected = self.E(k_shared)

        v_SA_projected = self.F(v_SA)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)
       
        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)
        
        # Concat fusion
        x_SA = self.out_proj(x_SA)
        
        x_CA = self.out_proj2(x_CA)
        
        x = torch.cat((x_SA, x_CA), dim=-1)
       
        x = x.transpose(-1,-2).reshape(b1,c1,h1,w1)
        
        x = x.permute(0, 3, 2, 1)
        
        x = x + x1
        return x

