import torch
import torch.nn as nn


class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = self.fc_x(x) + self.fc_y(y)
        return x, y, output


class ConcatFusion(nn.Module):
    def __init__(self, args, input_dim=1024, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.auxi_fc = nn.Linear(input_dim, output_dim)

        self.mu_dul_backbone = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
        )
        self.logvar_dul_backbone = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
        )
        self.args = args

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)

        if self.args.pme:

            mu_dul = self.mu_dul_backbone(output)
            logvar_dul = self.logvar_dul_backbone(output)
            std_dul = (logvar_dul * 0.5).exp()
            #
            epsilon = torch.randn_like(std_dul)

            if self.training:
                output = mu_dul + epsilon * std_dul
            else:
                output = mu_dul

            out = self.fc_out(output)
            auxi_out = self.fc_out(output)
        else:
            mu_dul = torch.zeros_like(output)
            logvar_dul = self.logvar_dul_backbone(output)
            std_dul = (logvar_dul * 0.5).exp()
            #
            out = self.fc_out(output)
            auxi_out = self.fc_out(output)
        return x, y, out, auxi_out, mu_dul, std_dul


class ConcatFusion_Swin(nn.Module):
    def __init__(self, args,input_dim=1024* 2, output_dim=100):
        super(ConcatFusion_Swin, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

        self.mu_dul_backbone = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
        )
        self.logvar_dul_backbone = nn.Sequential(
            nn.Linear(input_dim,input_dim),
            nn.BatchNorm1d(input_dim),
        )
        self.args = args

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)

        if self.args.pme:

            mu_dul = self.mu_dul_backbone(output)
            logvar_dul = self.logvar_dul_backbone(output)
            std_dul = (logvar_dul * 0.5).exp()
            #
            epsilon = torch.randn_like(std_dul)

            if self.training:
                output = mu_dul + epsilon * std_dul
            else:
                output = mu_dul

            out = self.fc_out(output)
            auxi_out = self.fc_out(output)
        else:
            mu_dul = torch.zeros_like(output)
            logvar_dul = self.logvar_dul_backbone(output)
            std_dul = (logvar_dul * 0.5).exp()
            #
            out = self.fc_out(output)
            auxi_out = self.fc_out(output)
        return x, y, out, auxi_out, mu_dul, std_dul



class ConcatFusion_Vanilla(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(ConcatFusion_Vanilla, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.auxi_fc = nn.Linear(input_dim, output_dim)
        #
        # self.mu_dul_backbone = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024),
        # )
        # self.logvar_dul_backbone = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024),
        # )
        # self.args=args

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        out = self.fc_out(output)
        auxi_out = self.auxi_fc(output)

        return x, y, out, auxi_out


class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_film=True):
        super(FiLM, self).__init__()

        self.fc = nn.Linear(512 * 512, 512)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_film = x_film

    def forward(self, x, y):
        # if self.x_film:
        #     film = x
        #     to_be_film = y
        # else:
        #     film = y
        #     to_be_film = x
        #
        # gamma, beta = torch.split(self.fc(film), self.dim, 1)
        #
        # output = gamma * to_be_film + beta
        x = torch.unsqueeze(x, dim=2)
        y = torch.unsqueeze(y, dim=1)
        z = torch.bmm(x, y)
        # print(z.shape)
        z = z.view(z.shape[0], -1)
        output = self.fc(z)
        output = self.fc_out(output)

        return x, y, output


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_gate=True):
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return out_x, out_y, output


