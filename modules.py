import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())



### edited selfAttention.... modified to deal with data shape batch,45,1
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm(channels)
        self.ff_self = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # added by Farzana to resolve the error
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.permute(0, 2, 1)

    
"""
## edited to conv1d for energy network       
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()
        self.residual = residual
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )
    
    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)
"""
## renewed doubleconv 12/02/2025
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False, dropout=0.0, use_bn=False):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels) if use_bn else nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),  # Apply dropout if enabled
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels) if use_bn else nn.GroupNorm(1, out_channels),
        )
        self.residual = residual and in_channels == out_channels

    def forward(self, x):
        out = self.double_conv(x)
        return F.gelu(x + out) if self.residual else out
#edited for energy network        
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=64,dropout=0.0, use_bn=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.MaxPool1d(2,padding=1), #padding=1 was not given in previous implementation, added to avoid L_out=0
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )
    
    def forward(self, x, emb):
        x = self.conv(x)
        # print('after conv before emb_layer in DownBlock',x.shape)
        # print('shape of emb',emb.shape)
        emb=self.emb_layer(emb)
        # print("emb shape now: ",emb.shape)
        # #emb = self.emb_layer(emb).unsqueeze(-1).expand(-1, -1, x.shape[-1])
        # emb = emb.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        emb = emb.mean(dim=1)  # Reduce from (4000, 45, 128) → (4000, 128)
        emb = emb.unsqueeze(-1)  # Now: (4000, 128, 1), matching `x`
        return x + emb

#updated for energy network    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=64,dropout=0.0, use_bn=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(2*in_channels, in_channels, residual=True,dropout=dropout, use_bn=use_bn),
            DoubleConv(in_channels, out_channels,dropout=dropout, use_bn=use_bn),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )
    
    def forward(self, x, skip_x, emb):
        x = self.up(x)
        #print("before cat skip and x ------->",skip_x.shape, x.shape)
        #added to remove the error
        if skip_x.shape[-1] != x.shape[-1]:
            skip_x = F.interpolate(skip_x, size=x.shape[-1], mode="linear", align_corners=True)

        x = torch.cat([skip_x, x], dim=1)
        #print("after cat in up ", x.shape)
        x = self.conv(x)
        
        #print("after conv in up ",x.shape, emb.shape)
        #emb = self.emb_layer(emb).unsqueeze(-1).expand(-1, -1, x.shape[-1])
        #edited by Farzana
        emb=self.emb_layer(emb)
        emb = emb.mean(dim=1)
        emb = emb.unsqueeze(-1)
        # print("after emb layer")
        # print(emb.shape)
        # emb = self.emb_layer(emb).unsqueeze(-1)  # Shape: (4000, 128, 1, 1)
        # emb = emb.expand(-1, 45, -1, -1)  
        #emb = emb.expand(-1, -1, x.shape[-2], x.shape[-1])  # Match `x` shape
        #print("final shape of x and emb ",x.shape, emb.shape)
        return x + emb

#modified for energy network
class UNet(nn.Module):
    def __init__(self, c_in=45, c_out=45, time_dim=64,dropout=0.0, use_bn=False):
        super().__init__()
        #downsampling
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128, emb_dim=time_dim, dropout=dropout, use_bn=use_bn)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256, emb_dim=time_dim, dropout=dropout, use_bn=use_bn)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256, emb_dim=time_dim, dropout=dropout, use_bn=use_bn)
        self.sa3 = SelfAttention(256)
        
        #bottleneck
        self.bot = DoubleConv(256, 256)
        
        #upsampling
        self.up1 = Up(256, 128, emb_dim=time_dim, dropout=dropout, use_bn=use_bn)  #  Corrected✅
        self.sa4 = SelfAttention(128)
        self.up2 = Up(128, 64, emb_dim=time_dim, dropout=dropout, use_bn=use_bn)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(64, 64, emb_dim=time_dim, dropout=dropout, use_bn=use_bn)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv1d(64, c_out, kernel_size=1)
    
    def forward(self, x, t_emb):
        #print("before inc",x.shape)
        x1 = self.inc(x)
        #print("after inc before down1",x1.shape)
        x2 = self.down1(x1, t_emb)
        #print("after down1 before sa1",x2.shape)
        x2 = self.sa1(x2)
        #print("after sa1 before down2",x2.shape)
        x3 = self.down2(x2, t_emb)
        #print("after down2 before sa2",x3.shape)
        x3 = self.sa2(x3)
        #print("after sa2 before down3",x3.shape)
        x4 = self.down3(x3, t_emb)
        #print("after down3 before sa3",x4.shape)
        x4 = self.sa3(x4)
        #print("after sa3 before bot",x4.shape)
        x4 = self.bot(x4)
        #print("after bot before up1",x4.shape)
        x = self.up1(x4, x3, t_emb)
        #print("after up1 before sa4",x.shape)
        x = self.sa4(x)
        #print("after sa4 before up2",x.shape)
        x = self.up2(x, x2, t_emb)
        #print("after up2 before sa5",x.shape)
        x = self.sa5(x)
        #print("after sa5 before up3",x.shape)
        x = self.up3(x, x1, t_emb)
        #print("after up3 before sa6",x.shape)
        x = self.sa6(x)
        #print("after sa6 before outc",x.shape)
        x = nn.AdaptiveAvgPool1d(1)(x)
        #print("after adaptation: ",x.shape)
        ret=self.outc(x)
        #print("shape of return finally: ",ret.shape)
        return ret






#modified for energy network
class UNet_conditional(UNet):
    def __init__(self, c_in=45, c_out=45, time_dim=64,dropout=0, use_bn=False):
        super().__init__(c_in, c_out, time_dim,dropout=dropout, use_bn=use_bn)
    
    def forward(self, x, time_emb, cond_emb ):
        t_emb = time_emb + cond_emb
        #print("inside UNet: ",t_emb.shape, x.shape,time_emb.shape, cond_emb.shape)
        return super().forward(x, t_emb)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, delta=0, path="checkpoint.pth",doc=None):
        """
        Args:
            patience (int): How long to wait after last improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model,doc):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), os.path.join(doc.basedir, f"{self.path}"))  # Save best model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
