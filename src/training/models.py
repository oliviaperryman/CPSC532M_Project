import torch
from torch import nn


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(
        self, input_nc, output_nc, nf=64, norm_layer=nn.BatchNorm2d, use_dropout=False
    ):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        # add the innermost block
        unet_block = UnetSkipConnectionBlock(
            nf * 8,
            nf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
        )
        # print(unet_block)

        # add intermediate block with nf * 8 filters
        unet_block = UnetSkipConnectionBlock(
            nf * 8,
            nf * 8,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
        )
        unet_block = UnetSkipConnectionBlock(
            nf * 8,
            nf * 8,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
        )
        unet_block = UnetSkipConnectionBlock(
            nf * 8,
            nf * 8,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
        )

        # gradually reduce the number of filters from nf * 8 to nf.
        unet_block = UnetSkipConnectionBlock(
            nf * 4, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            nf * 2, nf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            nf, nf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )

        # add the outermost block
        self.model = UnetSkipConnectionBlock(
            output_nc,
            nf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
        )

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(
            input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1
            )
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=False,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=False,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
            nn.Sigmoid(),
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
