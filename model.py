import torch
import torch.nn as nn


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
class DoubleCov(nn.Module):
    def __init__(self, in_channel, out_channel_1, out_channel_2):
        super(DoubleCov, self).__init__()
        self.cov1 = nn.Conv2d(in_channels=in_channel, out_channels= out_channel_1, kernel_size=3, padding=1)
        self.cov2 = nn.Conv2d(in_channels=out_channel_1, out_channels = out_channel_2, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.cov1(x)
        x = self.cov2(x)
        return x


class Merge(nn.Module):
    def __init__(self):
        super(Merge, self).__init__()

    def forward(self, x1, x2):
        [size_1_h, size_1_w] = x1.shape[2:4]
        [size_2_h, size_2_w] = x2.shape[2:4]

        assert (size_1_h >= size_2_h and size_1_w >= size_2_w)
        start_h = (size_1_h - size_2_h) // 2
        start_w = (size_1_w - size_2_w) // 2

        cropped = x1[:,:,start_h:start_h + size_2_h, start_w:start_w + size_2_w]
        merged = torch.cat((cropped, x2), dim=1)
        return merged


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.down_sampling_cov = nn.ModuleList()
        self.down_pool = nn.MaxPool2d(2, 2)
        self.button_cov = nn.ModuleList()
        self.merge = Merge()
        self.up_sampling_cov = nn.ModuleList()
        self.up_sampling_tran_cov = nn.ModuleList()
        self.final_cov = nn.ModuleList()


        channels = [3, 64, 128, 256, 512, 1024]

        for i in range(len(channels) - 2):
            input_channel = channels[i]
            output_channel = channels[i + 1]
            # 3->64->64->128->128->256->256->512->512
            self.down_sampling_cov.append(DoubleCov(input_channel, output_channel, output_channel))

        # input 512, output 1024
        self.button_cov.append(nn.Conv2d(channels[-2], channels[-1], kernel_size=3, padding=1))
        # input 1024, output 512
        self.button_cov.append(nn.Conv2d(channels[-1], channels[-1], kernel_size=3, padding=1))

        rev_channels = channels[:0:-1]
        rev_channels.append(64)
        # rev_channels = [1024, 512, 256, 128, 64, 64]
        # final output channel 64
        for i in range(len(rev_channels) - 2):
            input_channel = rev_channels[i]
            output_channel1 = rev_channels[i + 1]
            output_channel2= rev_channels[i + 2]
            self.up_sampling_cov.append(DoubleCov(input_channel, output_channel1, output_channel1))
            self.up_sampling_tran_cov.append(nn.ConvTranspose2d(input_channel, output_channel1, 2, 2))

        self.final_cov.append(nn.Conv2d(rev_channels[-1], 1, 3, padding=1))

    def forward(self, x):
        mid_features = []
        for i in range(len(self.down_sampling_cov)):
            x = self.down_sampling_cov[i](x)
            mid_features.append(x)
            x = self.down_pool(x)
        x = self.button_cov[0](x)
        x = self.button_cov[1](x)
        for i in range(len(self.up_sampling_tran_cov)):
            x = self.up_sampling_tran_cov[i](x)
            x = self.merge(mid_features[-1], x)
            mid_features.pop(-1)
            x = self.up_sampling_cov[i](x)
        x = self.final_cov[0](x)
        return x


if __name__ == "__main__":
    input = torch.randn(1,3,1024,768, device=DEVICE)
    model = Unet()
    model.to(device=DEVICE)
    output = model(input)
    print(output.shape)