import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        # self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        # coords_h = torch.arange(self.window_size[0])
        # coords_w = torch.arange(self.window_size[1])
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        # relative_coords[:, :, 1] += self.window_size[1] - 1
        # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        # attn = (q @ k.transpose(-2, -1))
        attn = (q.transpose(-2, -1) @ k)

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # attn = attn + relative_position_bias.unsqueeze(0)

        # if mask is not None:
        #     nW = mask.shape[0]
        #     attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        #     attn = attn.view(-1, self.num_heads, N, N)
        #     attn = self.softmax(attn)
        # else:
        #     attn = self.softmax(attn)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = (v @ attn).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads,  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # self.input_resolution = input_resolution
        self.num_heads = num_heads
        # self.window_size = window_size
        # self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # if min(self.input_resolution) <= self.window_size:
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = 0
        #     self.window_size = min(self.input_resolution)
        # assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # if self.shift_size > 0:
        #     attn_mask = self.calculate_mask(self.input_resolution)
        # else:
        #     attn_mask = None

        # self.register_buffer("attn_mask", attn_mask)

    # def calculate_mask(self, x_size):
    #     # calculate attention mask for SW-MSA
    #     H, W = x_size
    #     img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
    #     h_slices = (slice(0, -self.window_size),
    #                 slice(-self.window_size, -self.shift_size),
    #                 slice(-self.shift_size, None))
    #     w_slices = (slice(0, -self.window_size),
    #                 slice(-self.window_size, -self.shift_size),
    #                 slice(-self.shift_size, None))
    #     cnt = 0
    #     for h in h_slices:
    #         for w in w_slices:
    #             img_mask[:, h, w, :] = cnt
    #             cnt += 1
    #
    #     mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
    #     mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
    #     attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    #     attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    #
    #     return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        # if self.shift_size > 0:
        #     shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        # else:
        #     shifted_x = x

        # partition windows
        # x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x.view(-1, H*W, C)  # B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        # if self.input_resolution == x_size:
        #     attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        # else:
        #     attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))
        attn_windows = self.attn(x_windows)

        # merge windows
        # attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        # if self.shift_size > 0:
        #     x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        # else:
        #     x = shifted_x
        x = attn_windows

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicBlock(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        # self.input_resolution = input_resolution
        self.depth = depth
        # self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        # if downsample is not None:
        #     self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        # else:
        #     self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            x = blk(x, x_size)
        return x


class RTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, depth, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False, resi_connection='1conv'):
        super(RTB, self).__init__()

        self.dim = dim
        # self.input_resolution = input_resolution

        self.residual_group = BasicBlock(dim=dim,
                                         depth=depth,
                                         num_heads=num_heads,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
        #     norm_layer=None)
        #
        # self.patch_unembed = PatchUnEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
        #     norm_layer=None)

    def forward(self, x, x_size):
        H, W = x_size
        B = x.shape[0]
        shortcut = x
        x = self.residual_group(x, x_size)  # B, H*W, C
        x = x.flatten(2).transpose(1, 2).view(B, -1, H, W)    # B, C, H, W
        x = self.conv(x).flatten(2).transpose(1, 2) + shortcut
        return x


class SwinIR(nn.Module):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, in_chans=512, embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
                 **kwargs):
        super(SwinIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        # patches_resolution = self.patch_embed.patches_resolution
        # self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        # self.patch_unembed = PatchUnEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        # if self.ape:
        #     self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        #     trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RTB(dim=embed_dim,
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         resi_connection=resi_connection
                        )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # #####################################################################################################
        # ################################ 3, high quality image reconstruction ################################
        # if self.upsampler == 'pixelshuffle':
        #     # for classical SR
        #     self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
        #                                               nn.LeakyReLU(inplace=True))
        #     self.upsample = Upsample(upscale, num_feat)
        #     self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        # elif self.upsampler == 'pixelshuffledirect':
        #     # for lightweight SR (to save parameters)
        #     self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
        #                                     (patches_resolution[0], patches_resolution[1]))
        # elif self.upsampler == 'nearest+conv':
        #     # for real-world SR (less artifacts)
        #     assert self.upscale == 4, 'only support x4 now.'
        #     self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
        #                                               nn.LeakyReLU(inplace=True))
        #     self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        #     self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        #     self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        #     self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        #     self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # else:
        #     # for image denoising and JPEG compression artifact reduction
        #     self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # def check_image_size(self, x):
    #     _, _, h, w = x.size()
    #     mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
    #     mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
    #     x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    #     return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        H, W = x_size
        # x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        # if self.ape:
        #     x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        B = x.shape[0]
        # x = self.patch_unembed(x, x_size)

        return x.transpose(1, 2).view(B, -1, H, W)

    def forward(self, x):
        H, W = x.shape[2:]
        # x = self.check_image_size(x)

        # self.mean = self.mean.type_as(x)
        # x = (x - self.mean) * self.img_range

        # if self.upsampler == 'pixelshuffle':
        #     # for classical SR
        #     x = self.conv_first(x)
        #     x = self.conv_after_body(self.forward_features(x)) + x
        #     x = self.conv_before_upsample(x)
        #     x = self.conv_last(self.upsample(x))
        # elif self.upsampler == 'pixelshuffledirect':
        #     # for lightweight SR
        #     x = self.conv_first(x)
        #     x = self.conv_after_body(self.forward_features(x)) + x
        #     x = self.upsample(x)
        # elif self.upsampler == 'nearest+conv':
        #     # for real-world SR
        #     x = self.conv_first(x)
        #     x = self.conv_after_body(self.forward_features(x)) + x
        #     x = self.conv_before_upsample(x)
        #     x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
        #     x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
        #     x = self.conv_last(self.lrelu(self.conv_hr(x)))
        # else:
        #     # for image denoising and JPEG compression artifact reduction
        #     x_first = self.conv_first(x)
        #     res = self.conv_after_body(self.forward_features(x_first)) + x_first
        #     x = x + self.conv_last(res)
        x_first = self.conv_first(x)
        res = self.conv_after_body(self.forward_features(x_first)) + x_first
        x = x + self.conv_last(res)

        # x = x / self.img_range + self.mean

        return x
# a = torch.randn((1, 128, 64, 64)).cuda(0)
# model_edsr = SwinIR(in_chans=128, embed_dim=128, depths=[2, 2, 4, 2], num_heads=[4, 4, 4, 4]).cuda(0)
#
# def get_parameter_number(model_edsr):
#     total_num = sum(p.numel() for p in model_edsr.parameters())
#     trainable_num = sum(p.numel() for p in model_edsr.parameters() if p.requires_grad)
#     return {'Total': total_num, 'Trainable': trainable_num}
#
#
# print(get_parameter_number(model_edsr))

# print(model_edsr(a).shape)