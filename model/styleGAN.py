from .model_base import *


class MappingNet(nn.Module):
    def __init__(self,
                 z_dim,
                 w_dim,
                 normalize_latent=True,
                 use_leaky=True,
                 negative_slope=0.2,
                 equalize_lr=True,
                 gain=2 ** 0.5):
        super(MappingNet, self).__init__()
        self.mapping = []
        if normalize_latent:
            self.mapping.append(PixelNormLayer())
        act_fcn = nn.LeakyReLU(negative_slope) if use_leaky else nn.ReLU()
        if equalize_lr:
            self.mapping.extend([
                EqualizedLinear(z_dim, w_dim, gain), act_fcn,
                EqualizedLinear(w_dim, w_dim, gain), act_fcn,
                EqualizedLinear(w_dim, w_dim, gain), act_fcn,
                EqualizedLinear(w_dim, w_dim, gain), act_fcn,
                EqualizedLinear(w_dim, w_dim, gain), act_fcn,
                EqualizedLinear(w_dim, w_dim, gain), act_fcn,
                EqualizedLinear(w_dim, w_dim, gain), act_fcn,
                EqualizedLinear(w_dim, w_dim, gain), act_fcn
            ])
        else:
            self.mapping.extend([
                nn.Linear(z_dim, w_dim), act_fcn,
                nn.Linear(w_dim, w_dim), act_fcn,
                nn.Linear(w_dim, w_dim), act_fcn,
                nn.Linear(w_dim, w_dim), act_fcn,
                nn.Linear(w_dim, w_dim), act_fcn,
                nn.Linear(w_dim, w_dim), act_fcn,
                nn.Linear(w_dim, w_dim), act_fcn,
                nn.Linear(w_dim, w_dim), act_fcn
            ])
        self.mapping = nn.Sequential(*self.mapping)

    def forward(self, x):
        return self.mapping(x)


class SynthesisNet(nn.Module):
    def __init__(self,
                 num_channels=3,
                 z_dim=512,
                 w_dim=512,
                 max_resolution=256,
                 feature_map_base=4096,
                 feature_map_decay=1,
                 max_feature_maps=512,
                 use_leaky=True,
                 negative_slope=0.2,
                 equalize_lr=True):
        super(SynthesisNet, self).__init__()
        self.R = int(math.log2(max_resolution))
        assert 2 ** self.R == max_resolution
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.feature_map_base = feature_map_base
        self.feature_map_decay = feature_map_decay
        self.max_feature_maps = max_feature_maps
        self.start_constant_tensor = nn.Parameter(torch.ones(size=(1, max_feature_maps, 4, 4)))

        """Network Structure"""
        self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')
        self.to_rgb = nn.ModuleList()
        self.progressive_layers = nn.ModuleList()
        self.progressive_layers.append(GConvBlock(
            max_feature_maps, max_feature_maps, input_constant=True, kernel_size=3, stride=1, padding=1,
            use_leaky=use_leaky, negative_slope=negative_slope, w_dim=w_dim, equalize_lr=equalize_lr
        ))
        if equalize_lr:
            self.to_rgb.append(EqualizedConv2d(max_feature_maps, num_channels, kernel_size=1, padding=0))
        else:
            self.to_rgb.append(nn.Conv2d(max_feature_maps, num_channels, kernel_size=1, padding=0))
        for r in range(3, self.R + 1):
            in_channels, out_channels = self.get_feature_map_number(r - 1), self.get_feature_map_number(r)
            self.progressive_layers.append(GConvBlock(
                in_channels, out_channels, input_constant=False, kernel_size=3, stride=1, padding=1,
                use_leaky=use_leaky, negative_slope=negative_slope, w_dim=w_dim, equalize_lr=equalize_lr
            ))
            if equalize_lr:
                self.to_rgb.append(EqualizedConv2d(out_channels, num_channels, kernel_size=1, padding=0))
            else:
                self.to_rgb.append(nn.Conv2d(out_channels, num_channels, kernel_size=1, padding=0))

    def get_feature_map_number(self, stage):
        return min(int(self.feature_map_base / (2.0 ** (stage * self.feature_map_decay))), self.max_feature_maps)

    def forward(self, w, level, mode='stabilize', alpha=None):
        x = self.start_constant_tensor.expand(w.shape[0], -1, -1, -1)
        if mode == 'stabilize':
            from_, to_ = 0, level - 1
            for i in range(from_, to_):
                if i > 0:
                    x = self.upsampler(x)
                x = self.progressive_layers[i](x, w)
            x = self.to_rgb[to_ - 1](x)
            return x
        assert alpha is not None
        from_, to_ = 0, level - 2
        for i in range(from_, to_):
            if i > 0:
                x = self.upsampler(x)
            x = self.progressive_layers[i](x, w)
        out1 = self.upsampler(x)
        out1 = self.to_rgb[to_ - 1](out1)
        x = self.progress_growing[to_](x)
        out2 = self.to_rgb[to_](x)
        out = (1 - alpha) * out1 + alpha * out2
        return out


class Generator(nn.Module):
    def __init__(self,
                 num_channels=3,
                 z_dim=512,
                 w_dim=512,
                 normalize_latent=True,
                 max_resolution=256,
                 feature_map_base=4096,
                 feature_map_decay=1,
                 max_feature_maps=512,
                 use_leaky=True,
                 negative_slope=0.2,
                 equalize_lr=True,
                 **kwargs):
        super(Generator, self).__init__()
        self.mapping = MappingNet(z_dim, w_dim, normalize_latent,
                                  use_leaky=use_leaky, negative_slope=negative_slope,
                                  equalize_lr=equalize_lr, gain=2 ** 0.5)
        self.synthesis_net = SynthesisNet(num_channels, z_dim, w_dim, max_resolution,
                                          feature_map_base, feature_map_decay, max_feature_maps,
                                          use_leaky=use_leaky, negative_slope=negative_slope, equalize_lr=equalize_lr)

    def forward(self, z, level, mode='stabilize', alpha=None):
        w = self.mapping(z)
        x = self.synthesis_net(w, level=level, mode=mode, alpha=alpha)
        return x


class Discriminator(nn.Module):
    def __init__(self,
                 num_channels=3,
                 max_resolution=256,
                 feature_map_base=4096,
                 feature_map_decay=1.0,
                 max_feature_maps=512,
                 use_leaky=True,
                 negative_slope=0.2,
                 minibatch_stat_concat=True,
                 equalize_lr=True,
                 sigmoid_at_end=False,
                 **kwargs):
        super(Discriminator, self).__init__()
        self.feature_map_base = feature_map_base
        self.feature_map_decay = feature_map_decay
        self.max_feature_maps = max_feature_maps

        out_active = nn.Sigmoid() if sigmoid_at_end else None
        self.R = int(math.log2(max_resolution))
        assert 2 ** self.R == max_resolution

        """Network Structure"""
        act_fcn = nn.LeakyReLU(negative_slope) if use_leaky else nn.ReLU()
        self.from_rgb = nn.ModuleList()
        self.progressive_layers = nn.ModuleList()
        for r in range(self.R - 1, 1, -1):
            in_channels, out_channels = self.get_feature_map_number(r), self.get_feature_map_number(r - 1)
            if equalize_lr:
                self.from_rgb.append(nn.Sequential(
                    EqualizedConv2d(num_channels, in_channels, kernel_size=1, padding=0),
                    act_fcn
                ))
            else:
                self.from_rgb.append(nn.Sequential(
                    nn.Conv2d(num_channels, in_channels, kernel_size=1, padding=0),
                    act_fcn
                ))
            self.progressive_layers.append(
                nn.Sequential(
                    DConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                               use_leaky=use_leaky, negative_slope=negative_slope, equalize_lr=equalize_lr),
                    DConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                               use_leaky=use_leaky, negative_slope=negative_slope, equalize_lr=equalize_lr),
                    nn.AvgPool2d(kernel_size=2, stride=2, count_include_pad=False)
                )
            )
        last_layers = []
        in_channels, out_channels = self.get_feature_map_number(1), self.get_feature_map_number(1)
        if equalize_lr:
            self.from_rgb.append(nn.Sequential(
                EqualizedConv2d(num_channels, in_channels, kernel_size=1, padding=0),
                act_fcn
            ))
        else:
            self.from_rgb.append(nn.Sequential(
                nn.Conv2d(num_channels, in_channels, kernel_size=1, padding=0),
                act_fcn
            ))
        if minibatch_stat_concat:
            last_layers.append(MinibatchStatConcatLayer())
            in_channels += 1
        last_layers.extend([
            DConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                       use_leaky=use_leaky, negative_slope=negative_slope, equalize_lr=equalize_lr),
            DConvBlock(out_channels, self.get_feature_map_number(0), kernel_size=4, stride=1, padding=0,
                       use_leaky=use_leaky, negative_slope=negative_slope, equalize_lr=equalize_lr),
            Flatten()
        ])
        if equalize_lr:
            last_layers.append(EqualizedLinear(self.get_feature_map_number(0), 1, gain=1))
        else:
            last_layers.append(nn.Linear(self.get_feature_map_number(0), 1))
        if sigmoid_at_end:
            last_layers.append(out_active)
        self.progressive_layers.append(nn.Sequential(*last_layers))

    def get_feature_map_number(self, stage):
        return min(int(self.feature_map_base / (2.0 ** (stage * self.feature_map_decay))), self.max_feature_maps)

    def forward(self, x, level, mode='stabilize', alpha=None):
        """
        level: 表示正在进行的分辨率的2底数对数,例如,当前为64 pixel时,level为6
        mode: 取值为'stabilize'或'transition',后者在当前level进行fade in
        """
        # assert level in range(2, self.R + 1)
        # assert mode in {'stabilize', 'transition'}
        if mode == 'stabilize':
            from_, to_ = self.R - level, self.R - 1
            x = self.from_rgb[from_](x)
            for i in range(from_, to_):
                x = self.progressive_layers[i](x)
            return x
        assert alpha is not None
        from_, to_ = self.R - level + 1, self.R - 1
        in1 = F.avg_pool2d(x, kernel_size=2, stride=2)
        in1 = self.from_rgb[from_](in1)
        in2 = self.from_rgb[from_ - 1](x)
        in2 = self.progressive_layers[from_ - 1](in2)
        out = (1 - alpha) * in1 + alpha * in2
        for i in range(from_, to_):
            out = self.progressive_layers[i](out)
        return out
