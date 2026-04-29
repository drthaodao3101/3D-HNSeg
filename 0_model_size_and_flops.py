import torch 
from fvcore.nn import FlopCountAnalysis, parameter_count_table


t1 = torch.rand(1, 1, 96, 96, 96).cuda()
                                                       

from model_brain_mamba.Brain_Mamba import Brain_Mamba
model = Brain_Mamba(
        in_channels = 1,                                #n_channels
        n_channels = 48,                                #Feature size default for SwinTransformer
        n_classes = 4,                                  #out_classes 
        kernel_sizes=[1,3,5],
        strides=[1,1,1],
        uniform_dec_channels=48,                        #Defaut number of channels in each satge of the decoder
        deep_supervision=False,              
        do_res=True,                     
        do_res_up_down = True,
        block_counts = [2,2,2,2,2,2,2,2,2],              #[3,4,4,4,4,4,4,4,3]
        checkpoint_style = None,
        mode='train'                                    #unused
    ).cuda()


model.eval()

# -------- Params --------
params = sum(p.numel() for p in model.parameters())

# -------- FLOPs --------
flops = FlopCountAnalysis(model, t1).total()

print(f"Model Parameters: {params / 1e6:.5f} M")
print(f"Model FLOPs: {flops / 1e9:.5f} GFLOPs")




