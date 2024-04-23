from .Network import STN_Flow_relative as SFR
from .UNet import UNetFlow as UF
from .UNet import MyUNetFlow as MUF
from .UNet import MyUNetFlow2 as MUF2
import torch

def STN_Flow_relative(size, cuda_id=0):
    return SFR(size, cuda_id)


def UNetFlow(down_scales=5, output_scale=2,
             num_filters_base=128, max_filters=512,
             input_channels=128, output_channels=2,
             downsampling='pool', cuda_id=0, mode=None):
    # print(cuda_id)
    # input('##################')
    def wrap_multigpu(stn_module):
        if len([cuda_id]) > 0:
            assert (torch.cuda.is_available())
            stn_module.to(cuda_id)
            stn_module = torch.nn.DataParallel(stn_module, [cuda_id])  # multi-GPUs
        return stn_module
    if mode == 'MyUNetFlow':
        stn = MUF(down_scales=down_scales, output_scale=output_scale,
             num_filters_base=num_filters_base, max_filters=max_filters,
             input_channels=input_channels, output_channels=output_channels,
             downsampling=downsampling, cuda_id=cuda_id)
    elif mode == 'MyUNetFlow2':
        stn = MUF2(down_scales=down_scales, output_scale=output_scale,
             num_filters_base=num_filters_base, max_filters=max_filters,
             input_channels=input_channels, output_channels=output_channels,
             downsampling=downsampling, cuda_id=cuda_id)
    else:
        stn = UF(down_scales=down_scales, output_scale=output_scale,
                 num_filters_base=num_filters_base, max_filters=max_filters,
                 input_channels=input_channels, output_channels=output_channels,
                 downsampling=downsampling, cuda_id=cuda_id)
    return wrap_multigpu(stn)
