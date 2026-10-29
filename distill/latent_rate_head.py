"""Small learned latent-rate head before the frozen waveform decoder.

This operates on neural codec features, never on an output waveform. Same
frame count is an exact identity route. Base eight-step denoiser stays frozen.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LatentRateHead(nn.Module):
    def __init__(self,hidden=128):
        super().__init__()
        self.input=nn.Conv1d(27,hidden,3,padding=1)
        self.blocks=nn.ModuleList([nn.Sequential(
            nn.Conv1d(hidden,hidden,3,padding=d,dilation=d),nn.GELU(),
            nn.Conv1d(hidden,hidden,1)) for d in [1,2,4,8,1,2]])
        self.output=nn.Conv1d(hidden,24,3,padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self,latent,target_frames):
        batch,channels,source_frames=latent.shape
        target=target_frames[0]
        fine=latent.reshape(batch,24,6,source_frames).permute(0,1,3,2).reshape(batch,24,source_frames*6)
        resized=F.interpolate(fine,size=[target*6],mode='linear',align_corners=False)
        positions=torch.arange(target*6,device=latent.device,dtype=latent.dtype)
        source_time=(positions+0.5)*(source_frames/target)-0.5
        sin=torch.sin(source_time*2*math.pi).reshape(1,1,-1).expand(batch,1,-1)
        cos=torch.cos(source_time*2*math.pi).reshape(1,1,-1).expand(batch,1,-1)
        rate=torch.ones_like(sin)*(source_frames/target)
        value=F.gelu(self.input(torch.cat([resized,sin,cos,rate],dim=1)))
        for block in self.blocks:value=value+0.2*block(value)
        gate=(target!=source_frames).to(latent.dtype)
        corrected=resized+gate*self.output(value)
        return corrected.reshape(batch,24,target,6).permute(0,1,3,2).reshape(batch,144,target)
