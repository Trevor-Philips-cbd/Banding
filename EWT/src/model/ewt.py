from model import common
# import common
import torch
import torch.nn as nn
import torch.nn.functional as F
# import scipy.io as sio
# from model.masknet import MaskBlock
# from masknet import MaskBlock
from model.MFAM import MFAM
# from MFAM import MFAM
# from model.newarch import NEWARCH
# from newarch import NEWARCH
# from MFAM import MFAM
# from model.swinir import SwinIR
# from swinir import SwinIR
# from model.uformer import Uformer
# from uformer import Uformer
# from model.mwt import MWT
# from mwt import MWT
# from model.restormer import Restormer
# from restormer import Restormer
import os

def make_model(args, parent=False):
    return EWT(args)

class EWT(nn.Module):
    # def __init__(self, args, conv=common.default_conv):
    def __init__(self, conv=common.default_conv):
        super(EWT, self).__init__()
        self.scale_idx = 0

        self.DWT = common.DWT()
        self.IWT = common.IWT()
        # gray-4
        self.trans = MFAM(upscale=1, img_size=(32, 32), in_chans=12,
                     window_size=8, img_range=1., depths=[2, 2, 4],
                     embed_dim=96, num_heads=[6, 6, 6], mlp_ratio=2, upsampler='')

        # gray-1
        # self.trans = MFAM(upscale=1, img_size=(32, 32), in_chans=12,
        #                   window_size=8, img_range=1., depths=[2, 2, 4],
        #                   embed_dim=48, num_heads=[6, 6, 6], mlp_ratio=2, upsampler='')


        # self.trans = NEWARCH(upscale=1, img_size=(32, 32), in_chans=12,
        #              window_size=8, img_range=1., depths=[4, 4, 4, 4, 4],
        #              embed_dim=180, num_heads=[6, 6, 6, 6, 6], mlp_ratio=2, upsamplermpler='')
        # self.trans = SwinIR(upscale=1, img_size=(8, 8),
        #              window_size=8, img_range=1., depths=[6, 6, 6],
        #              embed_dim=180, num_heads=[6, 6, 6], mlp_ratio=2, upsampler='')
        # self.trans = Uformer(img_size=[64, 64], embed_dim=16, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        #         win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True,
        #         shift_flag=False)
        # self.trans = Restormer()
        # self.trans = MWT()
        # self.trans = MaskBlock()


    def _padding(self, x, scale):
        delta_H = 0
        delta_W = 0
        if x.shape[2] % scale != 0:
            delta_H = scale - x.shape[2] % scale
            x = F.pad(x, (0, 0, 0, delta_H), 'reflect')
        if x.shape[3] % scale != 0:
            delta_W = scale - x.shape[3] % scale
            x = F.pad(x, (0, delta_W, 0, 0), 'reflect')
        return x, delta_H, delta_W

    def _padding_2(self, x):
        _, _, H, W = x.shape
        delta = abs(H-W)
        if H < W:
            x = F.pad(x, (0, 0, 0, delta), 'reflect')
        elif H > W:
            x = F.pad(x, (0, delta, 0, 0), 'reflect')
        return x


    def forward(self, x):
        _, _, H, W = x.shape
        # x = self._padding_2(x)
        x, delta_H, delta_W = self._padding(x, 2)
        # print(x.shape)
        x = self.DWT(x)
        # x = self.DWT(x)
        x = self.trans(x)
        # x = self.IWT(x)
        # x = self.IWT(x)
        x = self.IWT(x)

        x = x[:, :, :H, :W]

        return x

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # input size: [batch_size, C, N], where C is number of dimension, N is the number of mesh.
    x = torch.rand(2, 3, 64, 64)
    x = x.cuda()
    # x = x.cuda()
    model = EWT()
    # model = model.cuda()
    # y = model(x)
    # print(y.shape)

    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    print(get_parameter_number(model))