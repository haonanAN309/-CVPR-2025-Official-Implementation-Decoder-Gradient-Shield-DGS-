import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()

        self.conv1x1_1 = nn.Conv2d(32, 32, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(32, 32, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(32, 32, kernel_size=1)

        self.conv3x3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # padding to keep dimensions

        self.conv5x5 = nn.Conv2d(32, 32, kernel_size=5, padding=2)  # padding to keep dimensions

        self.concat_conv1x1 = nn.Conv2d(96, 32, kernel_size=1)


    def forward(self, x):
        residual = x

        path1 = F.relu(self.conv1x1_1(x))

        path2 = F.relu(self.conv1x1_2(x))
        path2 = F.relu(self.conv3x3(path2))

        path3 = F.relu(self.conv1x1_3(x))
        path3 = F.relu(self.conv5x5(path3))

        concatenated = torch.cat((path1, path2, path3), dim=1)

        concatenated = F.relu(self.concat_conv1x1(concatenated))

        out = residual + concatenated

        return out


class EA(nn.Module):
    def __init__(self):
        super(EA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.inception1 = Inception()
        self.inception2 = Inception()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.inception1(x))
        x = F.relu(self.inception2(x))
        x = torch.tanh(self.conv2(x))
        return x


class EB(nn.Module):
    def __init__(self):
        super(EB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=48, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        return x


class EC(nn.Module):
    def __init__(self):
        super(EC, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=49, out_channels=48, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        return x


class HidingENet(nn.Module):
    def __init__(self):
        super(HidingENet, self).__init__()
        self.ea = EA()
        self.eb = EB()
        self.ec = EC()

    def forward(self, wm_img, key):
        wm_img = self.ea(wm_img)
        key = self.eb(key)
        concat = torch.cat((wm_img, key), dim=1)
        out = self.ec(concat)
        return out
