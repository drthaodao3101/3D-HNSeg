import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class MultiDilationDepthwiseConv3D(nn.Module):
    def __init__(self, in_channels, conv, kernel_sizes=[1,3,5], strides=[1,1,1], dw_parallel=True):
        super(MultiDilationDepthwiseConv3D, self).__init__()
        self.in_channels = in_channels
        self.dw_parallel = dw_parallel
        self.dilations = [(kernel - 1) // 2 if kernel > 1 else 1 for kernel in kernel_sizes]
        modified_kernel_sizes = [3 if kernel > 1 else 1 for kernel in kernel_sizes]
        #print(kernel_sizes, strides)

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                conv(self.in_channels, self.in_channels, modified_kernel_sizes[i], strides[i], kernel_sizes[i] // 2, dilation=self.dilations[i], groups=self.in_channels, bias=False),
            )
            for i in range(len(modified_kernel_sizes))
        ])

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x+dw_out
        # concatenate the features
        out = torch.cat(outputs, dim=1)
        return out

#--------------------------------------------------------------------------------------
class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))        # beta
        self.bias = nn.Parameter(torch.zeros(normalized_shape))         # gamma
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x, dummy_tensor=False):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x
        
class LayerNorm3D(nn.Module):                                                                           #https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """
    LayerNorm for 3D inputs supporting:
    - channels_last : (B, D, H, W, C)
    - channels_first: (B, C, D, H, W)

    This implementation follows ConvNeXt-style channel-first LayerNorm
    to avoid expensive permute operations.
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format

        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError

        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            # x: [B, D, H, W, C]
            return F.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            )

        elif self.data_format == "channels_first":
            # x: [B, C, D, H, W]
            mean = x.mean(dim=1, keepdim=True)
            var = (x - mean).pow(2).mean(dim=1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)

            # broadcast over D, H, W
            return (
                self.weight[:, None, None, None] * x
                + self.bias[:, None, None, None]
            )
        
class Linear(nn.Linear):
    def __init__(self, *args, channel_first=True, groups=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel_first = channel_first
        self.groups = groups

    def forward(self, x: torch.Tensor):
        if self.channel_first:
            if x.dim() == 5:  # B, C, D, H, W
                return F.conv3d(x, self.weight[:, :, None, None, None], self.bias, groups=self.groups)
            elif x.dim() == 4:  # B, C, H, W
                return F.conv2d(x, self.weight[:, :, None, None], self.bias, groups=self.groups)
            elif x.dim() == 3:  # B, C, L
                return F.conv1d(x,self.weight[:, :, None], self.bias, groups=self.groups)
            else:
                raise ValueError(f"Unsupported input dim: {x.dim()}")
        else:
            return F.linear(x, self.weight, self.bias)


class ToM_Block(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                nslices=num_slices,
        )
    
    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip
        
        return out

class ThreeD_Dilated_Block(nn.Module):

    def __init__(self, in_channels:int,d_conv:int=3,expand:int=1,conv_bias:bool=True, ):
        super().__init__()
        self.num_slices_dict={48:64, 96:32, 192:16, 384:8,768:4}
        self.in_channels=in_channels
        self.d_model=in_channels
        self.d_conv=d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.ln=Linear(in_features=self.d_model, out_features=self.d_inner)
        self.dwconv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            kernel_size=self.d_conv,
            padding=(self.d_conv - 1) // 2,
            bias=conv_bias,
        )
        #self.tom=ToM_Block(dim=self.d_inner,num_slices=self.num_slices_dict[self.d_inner])
        self.tom=ToM_Block(dim=self.d_inner,num_slices=1)
        
        #self.layer_norm=nn.LayerNorm(self.d_inner)
        self.layer_norm=LayerNorm(self.d_inner,data_format="channels_first")
        
        self.ln_skip=Linear(in_features=self.d_model, out_features=self.d_inner)
        
    def forward(self, x):
        x_skip= self.ln_skip(x)
        
        x=self.ln(x)
        x=self.dwconv3d(x)
        x=self.tom(x)
        #x=self.layer_norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        x=self.layer_norm(x)

        x=x*x_skip

        return x


class ThreeD_MixMLP_Block(nn.Sequential):                                                              #https://github.com/FrancescoSaverioZuppichini/SegFormer/blob/main/README.md
    def __init__(self, in_channels: int, expansion: int = 4):
        super().__init__(
            
            # Pointwise 1×1×1 convolution
            # Mixes information across channels only
            nn.Conv3d(in_channels, in_channels, kernel_size=1),

            # Depthwise 3×3×3 convolution
            # Performs local spatial mixing in D×H×W
            nn.Conv3d(
                in_channels,
                in_channels * expansion,
                kernel_size=3,
                padding=1,
                groups=in_channels,  # depthwise convolution
            ),

            # Non-linear activation
            nn.GELU(),

            # Pointwise 1×1×1 convolution
            # Projects features back to the original channel dimension
            nn.Conv3d(in_channels * expansion, in_channels, kernel_size=1),
        )
#--------------------------------------------------------------------------------------




class TD_MambaBlock(nn.Module):

    def __init__(self, 
                in_channels:int, 
                out_channels:int, 
                exp_r:int=4, 
                kernel_sizes=(1,3,5), 
                strides=(1,1,1),
                do_res:int=True,
                norm_type:str = 'group',
                dim = '3d',
                conv=None,
                grn = False,
                dilated_block_is_used=True,
                mixmlp_block_is_used=False,
                ):

        super().__init__()
        self.dilated_block_is_used=dilated_block_is_used
        self.mixmlp_block_is_used=mixmlp_block_is_used

        self.do_res = do_res
        self.in_channels = in_channels
        self.out_channels = out_channels
        exp_r = len(kernel_sizes)
        assert dim in ['2d', '3d']
        self.dim = dim
        if conv == None:
            if self.dim == '2d':
                conv = nn.Conv2d
            elif self.dim == '3d':
                conv = nn.Conv3d
            
        # First convolution layer with DepthWise Convolutions
        self.conv1 = MultiDilationDepthwiseConv3D(in_channels, conv, kernel_sizes=kernel_sizes, strides=strides, dw_parallel=True)

        # Normalization Layer. GroupNorm is used by default.
        if norm_type=='group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels, 
                num_channels=exp_r*in_channels
                )
        elif norm_type=='layer':
            self.norm = LayerNorm(
                normalized_shape=in_channels, 
                data_format='channels_first'
                )
        
        # GeLU activations
        self.act = nn.GELU()
        
        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv(
            in_channels = exp_r*in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
        
        if self.do_res and (self.in_channels != self.out_channels):
            self.res_conv = conv(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 1,
                padding = 0
            )

        self.grn = grn
        if grn:
            if dim == '3d':
                self.grn_beta = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1,1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1,1), requires_grad=True)
            elif dim == '2d':
                self.grn_beta = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1), requires_grad=True)

        if self.dilated_block_is_used:
            self.tddb=ThreeD_Dilated_Block(in_channels=out_channels)                                                                            #ThreeD Dilated Block


        if self.mixmlp_block_is_used:
            self.mixmlp=ThreeD_MixMLP_Block(in_channels=out_channels)                                                                           #ThreeD_MixMLP_Block

    def forward(self, x, dummy_tensor=None):
        
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.norm(x1))
        if self.grn:
            # gamma, beta: learnable affine transform parameters
            # X: input of shape (N,C,H,W,D)
            if self.dim == '3d':
                gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            elif self.dim == '2d':
                gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True)+1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)
        
        if self.dilated_block_is_used:
            x1=self.tddb(x1)                                                                                                                    #ThreeD Dilated Block

        if self.do_res:
            if self.in_channels != self.out_channels:
                x = self.res_conv(x)
            x1 = x + x1  

        if self.mixmlp_block_is_used:
            x1=self.mixmlp(x1)                                                                                                                  #ThreeD_MixMLP_Block
        
        return x1

class TD_MambaDownBlock(TD_MambaBlock):

    def __init__(self, in_channels, 
                 out_channels, 
                 exp_r:int=4, 
                 kernel_sizes=[1,3,5], 
                 strides=[2,1,1],
                 do_res=False, 
                 norm_type = 'group', 
                 dim='3d', 
                 grn=False,
                 dilated_block_is_used=False,
                 mixmlp_block_is_used=False,
                ):

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
        
        super().__init__(in_channels, out_channels, exp_r, kernel_sizes, strides=strides, 
                        do_res = False, norm_type = norm_type, dim=dim, grn=grn,
                        dilated_block_is_used=dilated_block_is_used,
                        mixmlp_block_is_used=mixmlp_block_is_used)
        self.resample_do_res = do_res
        if do_res:
            self.res_conv = conv(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 2
            )

    def forward(self, x, dummy_tensor=None):
        
        x1 = super().forward(x)
        
        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1


class TD_MambaUpBlock(TD_MambaBlock):

    def __init__(self, in_channels, 
                 out_channels, 
                 exp_r:int=4, 
                 kernel_sizes=[1,3,5], 
                 strides=[2,1,1],
                do_res=False, 
                norm_type = 'group', 
                dim='3d', 
                grn = False,
                dilated_block_is_used=False,
                mixmlp_block_is_used=False,
                ):

        self.resample_do_res = do_res
        
        self.dim = dim
        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d
        
        super().__init__(in_channels, out_channels, exp_r, kernel_sizes=kernel_sizes, strides=strides,
                    do_res=False, norm_type = norm_type, dim=dim, conv=conv,grn=grn,
                    dilated_block_is_used=dilated_block_is_used,
                    mixmlp_block_is_used=mixmlp_block_is_used) 
        if do_res:            
            self.res_conv = conv(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 2
                )

    def forward(self, x, dummy_tensor=None):
        
        x1 = super().forward(x)
        # Asymmetry but necessary to match shape
        if self.dim == '2d':
            x1 = torch.nn.functional.pad(x1, (1,0,1,0))
        elif self.dim == '3d':
            x1 = torch.nn.functional.pad(x1, (1,0,1,0,1,0))
        
        if self.resample_do_res:
            res = self.res_conv(x)
            if self.dim == '2d':
                res = torch.nn.functional.pad(res, (1,0,1,0))
            elif self.dim == '3d':
                res = torch.nn.functional.pad(res, (1,0,1,0,1,0))
            x1 = x1 + res

        return x1


class OutBlock(nn.Module):

    def __init__(self, in_channels, n_classes, dim, stride=1):
        super().__init__()
        
        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d
        self.conv_out = conv(in_channels, n_classes, kernel_size=1, stride = stride)
    
    def forward(self, x, dummy_tensor=None): 
        return self.conv_out(x)

class MLPBlock(nn.Module):                                  #https://github.com/aryateja2106/nested-learning/blob/main/src/core/memory.py
    """
    MLP block with residual connection.

    Architecture (Equation 89):
        M(x) = x + W1 * σ(W2 * x)
    """

    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden

        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "silu":
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following standard practice."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x

class Skip_Project(nn.Module):

    def __init__(self, 
                in_channels:int, 
                out_channels:int,
                dim = '3d',
                conv=None,
                ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        assert dim in ['2d', '3d']
        self.dim = dim
        if conv == None:
            if self.dim == '2d':
                conv = nn.Conv2d
            elif self.dim == '3d':
                conv = nn.Conv3d
            
        # First convolution layer with DepthWise Convolutions
        self.project_conv = conv(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 1,
                padding = 0
            )
        self.norm = LayerNorm(
            normalized_shape=out_channels, 
            data_format='channels_first'
            )
        # GeLU activations
        self.act = nn.GELU()
    
    def forward(self, x, dummy_tensor=None):
        x = self.project_conv(x)
        x = self.act(self.norm(x))
        return x


class Cross_Update_Block(nn.Module):

    def __init__(self, 
                in_channels:int, 
                out_channels:int,
                do_res:int=True,
                dim = '3d',
                conv=None,
                num_heads=3,
                attn_downsample=None,
                thresholds = [1.0, 0.7, 0.3]
                ):

        super().__init__()

        self.do_res = do_res
        self.attn_downsample=attn_downsample
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert dim in ['2d', '3d']
        self.dim = dim
        if conv == None:
            if self.dim == '2d':
                conv = nn.Conv2d
            elif self.dim == '3d':
                conv = nn.Conv3d
    
        if self.do_res and (self.in_channels != self.out_channels):
            self.res_conv = conv(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 1,
                padding = 0
            )

        self.cu_dim = out_channels
        self.cu_norm = nn.LayerNorm(self.cu_dim)
        self.num_heads = num_heads

        self.q_proj = nn.Linear(self.cu_dim, self.cu_dim)
        self.k_proj = nn.Linear(self.cu_dim, self.cu_dim)
        self.v_proj = nn.Linear(self.cu_dim, self.cu_dim)
        self.out_proj = nn.Linear(self.cu_dim, self.cu_dim)

        self.mem_low  = MLPBlock(self.cu_dim, self.cu_dim * 4)
        self.mem_mid  = MLPBlock(self.cu_dim, self.cu_dim * 4)
        self.mem_high = MLPBlock(self.cu_dim, self.cu_dim * 4)

        self.thresholds=thresholds
    
    def forward(self, x,y, dummy_tensor=None):
        
        x1 = x

        ###Cross-Update###
        if self.attn_downsample is not None:
            if self.dim == '3d':
                x1_ds = F.avg_pool3d(x1, kernel_size=self.attn_downsample, stride=self.attn_downsample)
                y_ds  = F.avg_pool3d(y,  kernel_size=self.attn_downsample, stride=self.attn_downsample)
            else:
                x1_ds = F.avg_pool2d(x1, kernel_size=self.attn_downsample, stride=self.attn_downsample)
                y_ds  = F.avg_pool2d(y,  kernel_size=self.attn_downsample, stride=self.attn_downsample)
        else:
            x1_ds=x1
            y_ds =y
        B, C = x1_ds.shape[:2]
        assert C == self.cu_dim
        n_x_tokens = x1_ds.shape[2:].numel()
        img_x_dims = x1_ds.shape[2:]
        #x_flat = x1_ds.reshape(B, C, n_x_tokens).transpose(-1, -2)            #B*N*C
        x_flat = x1_ds.flatten(2).transpose(1, 2)
        x_norm = self.cu_norm(x_flat)


        _, C = y_ds.shape[:2]
        assert C == self.cu_dim
        n_y_tokens = y_ds.shape[2:].numel()
        img_y_dims = y_ds.shape[2:]
        #y_flat = y_ds.reshape(B, C, n_y_tokens).transpose(-1, -2)            #B*N*C
        y_flat = y_ds.flatten(2).transpose(1, 2)
        y_norm = self.cu_norm(y_flat)


        Q = self.q_proj(x_norm)  # B N C
        K = self.k_proj(y_norm)  # B M C
        V = self.v_proj(y_norm)  # B M C

        B, N, C = Q.shape
        M = K.shape[1]
        H = self.num_heads
        D = C // H

        Q = Q.view(B, N, H, D).transpose(1, 2)  # B H N D
        K = K.view(B, M, H, D).transpose(1, 2)  # B H M D
        V = V.view(B, M, H, D).transpose(1, 2)  # B H M D
       
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False
        )

        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.out_proj(out)

        
        out = self.mem_low(out)
        out = self.mem_mid(out)
        out = self.mem_high(out)
            


        #token to vision
        out = out.transpose(1, 2)  # B, C, N
        out = out.reshape(B, C, *img_x_dims)  # B, C, ...
        if self.attn_downsample is not None:
            if self.dim == '3d':
                out = F.interpolate(out,size=x1.shape[2:],mode='trilinear',align_corners=False)
            else:
                out = F.interpolate(out,size=x1.shape[2:],mode='bilinear', align_corners=False)

        if self.do_res:
            if self.in_channels != self.out_channels:
                x = self.res_conv(x)
            out = x + out  
        return out


class TD_Mamba_Memory_Skip_Block(TD_MambaBlock):

    def __init__(self, in_channels, 
                 out_channels, 
                 exp_r:int=4, 
                 kernel_sizes=[1,3,5], 
                 strides=(1,1,1),
                 do_res=True, 
                 norm_type = 'group', 
                 dim='3d', 
                 grn=False,
                 num_heads=3,
                 dilated_block_is_used=False,
                 mixmlp_block_is_used=False,
                 thresholds = [0.3, 0.5, 0.7]
                ):

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
        
        super().__init__(in_channels, out_channels, exp_r, kernel_sizes, strides=strides, 
                        do_res = False, norm_type = norm_type, dim=dim, grn=grn,
                        dilated_block_is_used=dilated_block_is_used,
                        mixmlp_block_is_used=mixmlp_block_is_used)
        self.skip_do_res = do_res
        if do_res:
            self.res_conv = conv(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 1,
                padding = 0
            )

        self.cu_dim = out_channels
        self.cu_norm = nn.LayerNorm(self.cu_dim)
        
        self.thresholds = thresholds
        self.gate_proj = nn.Linear(self.cu_dim, 1)
        self.mem_low  = MLPBlock(self.cu_dim, self.cu_dim * 4)
        self.mem_mid  = MLPBlock(self.cu_dim, self.cu_dim * 4)
        self.mem_high = MLPBlock(self.cu_dim, self.cu_dim * 4)

    def forward(self, x, dummy_tensor=None):
        
        x1 = super().forward(x)
        
        B, C = x1.shape[:2]
        assert C == self.cu_dim
        n_x_tokens = x1.shape[2:].numel()
        img_x_dims = x1.shape[2:]
        x1 = x1.flatten(2).transpose(1, 2)
        x1 = self.cu_norm(x1)

        # Soft Nested Memory Fusion
        # -------------------------

        # Learnable score
        score = torch.sigmoid(self.gate_proj(x1))  # (B, N, 1)

        thr_low, thr_mid, thr_high = self.thresholds

        # Soft regions
        w_low  = torch.sigmoid(10 * (thr_low  - score))
        w_mid  = torch.sigmoid(10 * (score - thr_mid)) * torch.sigmoid(10 * (thr_high - score))
        w_high = torch.sigmoid(10 * (score - thr_high))

        # Normalize weights (giữ scale ổn định)
        weights = torch.cat([w_low, w_mid, w_high], dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)

        # Memory blocks
        low_out  = self.mem_low(x1)
        mid_out  = self.mem_mid(x1)
        high_out = self.mem_high(x1)

        # Weighted sum
        x1 = (
            weights[..., 0:1] * low_out +
            weights[..., 1:2] * mid_out +
            weights[..., 2:3] * high_out
        )

        #token to vision
        x1 = x1.transpose(1, 2)  # B, C, N
        x1 = x1.reshape(B, C, *img_x_dims)  # B, C, ...

        if self.skip_do_res:
            res = self.res_conv(x)
            x1 = x1 + res
        return x1  
     
if __name__ == "__main__":

    network = TDMambaBlock(in_channels=12, out_channels=12, do_res=True, grn=True, norm_type='group').cuda()
    # network = LayerNorm(normalized_shape=12, data_format='channels_last').cuda()
    # network.eval()
    with torch.no_grad():
        print(network)
        x = torch.zeros((2, 12, 64, 64, 64)).cuda()
        print(network(x).shape)
