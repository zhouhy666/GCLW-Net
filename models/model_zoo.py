from .unet_model import *
from .UNetpp import *
from .deeplabv3 import *
from .SegNet import *
from .PSPNet import *
from .BiSeNet import *
from .OCNet import *
MODEL_ZOO = {
    'UNet': UNet,
    'UNetRepLKBlock_Conv':UNetRepLKBlock_Conv,
    'UNetFastKAN':UNetFastKAN,
    'UNet_MobileNet': UNet_MobileNet,
    'UNet_MobileNet_Rep': UNet_MobileNet_Rep,
    'UNet_MobileNetV4': UNet_MobileNetV4,
    'ResUNet':ResUNet,
    'UNet_RepLKBlockConv_FastKAN':UNet_RepLKBlockConv_FastKAN,
    'UNet_ADown_RepLKBlockConv_FastKAN':UNet_ADown_RepLKBlockConv_FastKAN,
    'UNet_PConv_RepLKBlock_FastKAN': UNet_PConv_RepLKBlock_FastKAN,
    'UNetCBAM':UNetCBAM,
    'UNetSimAM':UNetSimAM,
    'UNetShuffleAttention':UNetShuffleAttention,
    'UNetCBAMaspp':UNetCBAMaspp,
    'UNet_star':UNet_star,
    'UNet_ema':UNet_ema,
    'UNet_PPA':UNet_PPA,
    'UNet_se':UNet_se,
    'UNet_secbam':UNet_secbam,
    'UNet_Dyse':UNet_Dyse,
    'UNet_proj_CBAM':UNet_proj_CBAM,
    'UNet_Dylian':UNet_Dylian,
    "UNet_ResPath":UNet_ResPath,
    "UNet_Proj":UNet_Proj,
    "UNetASPP":UNetASPP,
    "UNetASPP_CBAM":UNetASPP_CBAM,
    #对比
    "UNetPP":UNetPP,
    "DeepLabV3Plus":DeepLabV3Plus,
    "SegNet":SegNet,
    "BiSeNet":BiSeNet,
    "OCNet":OCNet,
    "PSPNet":PSPNet
}
