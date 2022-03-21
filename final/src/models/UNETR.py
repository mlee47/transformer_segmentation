import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)
        return x


class BaselineUNet(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super(BaselineUNet, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        self.block_1_1_left = BasicConv3d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_left = BasicConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)  # 64, 1/2
        self.block_2_1_left = BasicConv3d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = BasicConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)  # 128, 1/4
        self.block_3_1_left = BasicConv3d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = BasicConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)  # 256, 1/8
        self.block_4_1_left = BasicConv3d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left = BasicConv3d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_3_1_right = BasicConv3d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = BasicConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_2_1_right = BasicConv3d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = BasicConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_1 = nn.ConvTranspose3d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_1_1_right = BasicConv3d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = BasicConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.conv1x1 = nn.Conv3d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        ds0 = self.block_1_2_left(self.block_1_1_left(x))
        ds1 = self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0)))
        ds2 = self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1)))
        x = self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2)))

        x = self.block_3_2_right(self.block_3_1_right(torch.cat([self.upconv_3(x), ds2], 1)))
        x = self.block_2_2_right(self.block_2_1_right(torch.cat([self.upconv_2(x), ds1], 1)))
        x = self.block_1_2_right(self.block_1_1_right(torch.cat([self.upconv_1(x), ds0], 1)))

        x = self.conv1x1(x)
        if self.n_cls == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)


class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class SelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(embed_dim, self.all_head_size) #가중치들 학습!
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

        self.vis = False

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape) # (1,729,96) --> (1,729,12,8)
        return x.permute(0, 2, 1, 3) #(1,12,729,8)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, in_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1()
        x = self.act(x)
        x = self.drop(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=786, d_ff=2048, dropout=0.1):
        super().__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, dropout):
        super().__init__()
        self.n_patches = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv3d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (1,2,144,144,144) 입력
        x = self.patch_embeddings(x) # (1,2,144,144,144) --> (1,96,9,9,9)
        x = x.flatten(2) # (1,96,9,9,9) --> (1,96,729)
        x = x.transpose(-1, -2) # (1,96,729) --> (1,729,96)

        embeddings = x + self.position_embeddings #그대로
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, cube_size, patch_size):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_dim = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.mlp = PositionwiseFeedForward(embed_dim, 2048)
        self.attn = SelfAttention(num_heads, embed_dim, dropout)

    def forward(self, x):
        h = x #x.shape = (1,729,96)
        x = self.attention_norm(x) #x.shape = (1,729,96)
        x, weights = self.attn(x)
        x = x + h
        h = x

        x = self.mlp_norm(x)
        x = self.mlp(x)

        x = x + h
        return x, weights


class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, num_heads, num_layers, dropout, extract_layers):
        super().__init__()
        self.embeddings = Embeddings(input_dim, embed_dim, cube_size, patch_size, dropout)
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.extract_layers = extract_layers
        for _ in range(num_layers):
            layer = TransformerBlock(embed_dim, num_heads, dropout, cube_size, patch_size)
            self.layer.append(copy.deepcopy(layer))



    def forward(self, x):
        extract_layers = []
        hidden_states = self.embeddings(x)

        for depth, layer_block in enumerate(self.layer):
            hidden_states, _ = layer_block(hidden_states)
            if depth + 1 in self.extract_layers:
                extract_layers.append(hidden_states)

        return extract_layers


class ITN3D(nn.Module):
    def __init__(self, nf=16):
        super(ITN3D, self).__init__()

        self.conv0 = nn.Conv3d(1, nf, kernel_size=3, padding=1) #64-64
        self.bn0 = nn.BatchNorm3d(nf)
        self.conv1 = nn.Conv3d(nf, nf*2, kernel_size=3, padding=1, stride=2) #64-32
        self.bn1 = nn.BatchNorm3d(nf*2)
        self.conv2 = nn.Conv3d(nf*2, nf*4, kernel_size=3, padding=1, stride=2) #32-16
        self.bn2 = nn.BatchNorm3d(nf*4)
        self.conv3 = nn.Conv3d(nf * 4, nf * 8, kernel_size=3, padding=1, stride=2)  # 16-8
        self.bn3 = nn.BatchNorm3d(nf * 8)

        self.bottleneck0 = nn.Conv3d(nf*8, nf*8, kernel_size=3, padding=1) #8-8
        self.bnb0 = nn.BatchNorm3d(nf * 8)
        self.bottleneck1 = nn.Conv3d(nf*8, nf*8, kernel_size=3, padding=1) #8-8
        self.bnb1 = nn.BatchNorm3d(nf * 8)

        self.up31 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # 8-16
        self.pad3 = nn.ConstantPad3d(1, 0)
        self.up32 = nn.Conv3d(nf * 8, nf * 4, kernel_size=3, padding=0)
        self.drop3 = nn.Dropout(0.5)

        self.up21 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) #16-32
        self.pad2 = nn.ConstantPad3d(1, 0)
        self.up22 = nn.Conv3d(nf*4 + nf*4, nf*2, kernel_size=3, padding=0)
        self.drop2 = nn.Dropout(0.5)

        self.up11 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) #32-64
        self.pad1 = nn.ConstantPad3d(1, 0)
        self.up12 = nn.Conv3d(nf*2 + nf*2, nf, kernel_size=3, padding=0)
        self.drop1 = nn.Dropout(0.5)

        self.pad0 = nn.ConstantPad3d(1, 0)
        self.output = nn.Conv3d(nf + nf, 1, kernel_size=3, padding=0)

    def forward(self, x):

        c0 = F.relu(self.bn0(self.conv0(x)))
        c1 = F.relu(self.bn1(self.conv1(c0)))
        c2 = F.relu(self.bn2(self.conv2(c1)))
        c3 = F.relu(self.bn3(self.conv3(c2)))

        b0 = F.relu(self.bnb0(self.bottleneck0(c3)))
        b1 = F.relu(self.bnb1(self.bottleneck1(b0)))

        u3 = F.relu(self.up32(self.pad3(self.up31(b1))))
        u3cat = self.drop3(torch.cat([u3, c2], 1))
        u2 = F.relu(self.up22(self.pad2(self.up21(u3cat))))
        u2cat = self.drop2(torch.cat([u2, c1], 1))
        u1 = F.relu(self.up12(self.pad1(self.up11(u2cat))))
        u1cat = self.drop1(torch.cat([u1, c0], 1))
        out = self.output(self.pad0(u1cat)) + x

        return torch.sigmoid(out)


class UNETR(nn.Module):
    def __init__(self, img_shape=(144, 144, 144), input_dim=2, output_dim=1, embed_dim=96, patch_size=16, num_heads=12, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]

        self.patch_dim = [int(x / patch_size) for x in img_shape]

        # Transformer Encoder
        self.transformer = \
            Transformer(
                input_dim,
                embed_dim,
                img_shape,
                patch_size,
                num_heads,
                self.num_layers,
                dropout,
                self.ext_layers
            )

        # U-Net Decoder
        self.decoder0 = \
            nn.Sequential(
                Conv3DBlock(input_dim, 4, 3),
                Conv3DBlock(4, 8, 3)
            )

        self.decoder3 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, 64),
                Deconv3DBlock(64, 32),
                Deconv3DBlock(32, 16)
            )

        self.decoder6 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, 64),
                Deconv3DBlock(64, 32),
            )

        self.decoder9 = \
            Deconv3DBlock(embed_dim, 64)

        self.decoder12_upsampler = \
            SingleDeconv3DBlock(embed_dim, 64)

        self.decoder9_upsampler = \
            nn.Sequential(
                Conv3DBlock(128, 64),
                Conv3DBlock(64, 64),
                Conv3DBlock(64, 64),
                SingleDeconv3DBlock(64, 32)
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv3DBlock(64, 32),
                Conv3DBlock(32, 32),
                SingleDeconv3DBlock(32, 16)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv3DBlock(32, 16),
                Conv3DBlock(16, 16),
                SingleDeconv3DBlock(16, 8)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv3DBlock(16, 8),
                Conv3DBlock(8, 8),
                SingleConv3DBlock(8, output_dim, 1)
            )

    def forward(self, x):
        z = self.transformer(x)
        z0, z3, z6, z9, z12 = x, *z

        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))
        return output


class UNETR_ITN(nn.Module):
    # 이 모델이 proposed한 모델이고 다른 것들을 참고해서 구현했습니다.
    def __init__(self, img_shape=(144, 144, 144), input_dim=2, output_dim=1, embed_dim=96, patch_size=16, num_heads=12, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]

        self.patch_dim = [int(x / patch_size) for x in img_shape]

        self.attn_map = ITN3D()

        # Transformer Encoder
        self.transformer = \
            Transformer(
                input_dim,
                embed_dim,
                img_shape,
                patch_size,
                num_heads,
                self.num_layers,
                dropout,
                self.ext_layers
            )

        # U-Net Decoder
        self.decoder0 = \
            nn.Sequential(
                Conv3DBlock(input_dim, 4, 3),
                Conv3DBlock(4, 8, 3)
            )

        self.decoder3 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, 64),
                Deconv3DBlock(64, 32),
                Deconv3DBlock(32, 16)
            )

        self.decoder6 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, 64),
                Deconv3DBlock(64, 32),
            )

        self.decoder9 = \
            Deconv3DBlock(embed_dim, 64)

        self.decoder12_upsampler = \
            SingleDeconv3DBlock(embed_dim, 64)

        self.decoder9_upsampler = \
            nn.Sequential(
                Conv3DBlock(128, 64),
                Conv3DBlock(64, 64),
                Conv3DBlock(64, 64),
                SingleDeconv3DBlock(64, 32)
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv3DBlock(64, 32),
                Conv3DBlock(32, 32),
                SingleDeconv3DBlock(32, 16)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv3DBlock(32, 16),
                Conv3DBlock(16, 16),
                SingleDeconv3DBlock(16, 8)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv3DBlock(16, 8),
                Conv3DBlock(8, 8),
                SingleConv3DBlock(8, output_dim, 1)
            )

        self.avgpool1 = nn.AvgPool3d(8)
        self.avgpool2 = nn.AvgPool3d(4)
        self.avgpool3 = nn.AvgPool3d(2)

    def forward(self, x):
        y = x[:, 1, :, :, :]
        y = y.unsqueeze(1)
        attn = self.attn_map(y)
        attn_1 = self.avgpool1(attn)
        attn_2 = self.avgpool2(attn)
        attn_3 = self.avgpool3(attn)


        z = self.transformer(x)
        z0, z3, z6, z9, z12 = x, *z
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9*attn_1, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6*attn_2, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3*attn_3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0*attn, z3], dim=1))
        return output


class UNETR_ITN_Unet(nn.Module):
    def __init__(self, img_shape=(144, 144, 144), input_dim=2, output_dim=1, embed_dim=96, patch_size=16, num_heads=12, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]

        self.patch_dim = [int(x / patch_size) for x in img_shape]

        self.attn_map = BaselineUNet(1,1,16)

        # Transformer Encoder
        self.transformer = \
            Transformer(
                input_dim,
                embed_dim,
                img_shape,
                patch_size,
                num_heads,
                self.num_layers,
                dropout,
                self.ext_layers
            )

        # U-Net Decoder
        self.decoder0 = \
            nn.Sequential(
                Conv3DBlock(input_dim, 4, 3),
                Conv3DBlock(4, 8, 3)
            )

        self.decoder3 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, 64),
                Deconv3DBlock(64, 32),
                Deconv3DBlock(32, 16)
            )

        self.decoder6 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, 64),
                Deconv3DBlock(64, 32),
            )

        self.decoder9 = \
            Deconv3DBlock(embed_dim, 64)

        self.decoder12_upsampler = \
            SingleDeconv3DBlock(embed_dim, 64)

        self.decoder9_upsampler = \
            nn.Sequential(
                Conv3DBlock(128, 64),
                Conv3DBlock(64, 64),
                Conv3DBlock(64, 64),
                SingleDeconv3DBlock(64, 32)
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv3DBlock(64, 32),
                Conv3DBlock(32, 32),
                SingleDeconv3DBlock(32, 16)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv3DBlock(32, 16),
                Conv3DBlock(16, 16),
                SingleDeconv3DBlock(16, 8)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv3DBlock(16, 8),
                Conv3DBlock(8, 8),
                SingleConv3DBlock(8, output_dim, 1)
            )

        self.avgpool1 = nn.AvgPool3d(8)
        self.avgpool2 = nn.AvgPool3d(4)
        self.avgpool3 = nn.AvgPool3d(2)

    def forward(self, x):
        y = x[:, 1, :, :, :]
        y = y.unsqueeze(1)
        attn = self.attn_map(y)
        attn_1 = self.avgpool1(attn)
        attn_2 = self.avgpool2(attn)
        attn_3 = self.avgpool3(attn)


        z = self.transformer(x)
        z0, z3, z6, z9, z12 = x, *z
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9*attn_1, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6*attn_2, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3*attn_3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0*attn, z3], dim=1))
        return output

