"""Implements encoders for robotics tasks.
Code adapted from https://github.com/stanford-iprl-lab/multimodal_representation/blob/master/multimodal/models/base_models/encoders.py
"""
import torch.nn as nn


def init_weights(modules):
    """Weight initialization from original SensorFusion Code."""
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class CausalConv1D(nn.Conv1d):
    """Implements a causal 1D convolution."""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True
    ):
        """Initialize CausalConv1D Object.

        Args:
            in_channels (int): Input Dimension
            out_channels (int): Output Dimension
            kernel_size (int): Kernel Size
            stride (int, optional): Stride Amount. Defaults to 1.
            dilation (int, optional): Dilation Amount. Defaults to 1.
            bias (bool, optional): Whether to add a bias after convolving. Defaults to True.
        """
        self.__padding = (kernel_size - 1) * dilation

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        """Apply CasualConv1D layer to layer input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        res = super().forward(x)
        if self.__padding != 0:
            return res[:, :, : -self.__padding]
        return res


class ProprioEncoder(nn.Module):
    """Implements image encoder module.

    Sourced from selfsupervised code.
    """

    def __init__(self, z_dim, initialize_weights=True):
        """Initialize ProprioEncoder Module.

        Args:
            z_dim (float): Z dimension size
            initialize_weights (bool, optional): Whether to initialize weights or not. Defaults to True.
        """
        super().__init__()
        self.z_dim = z_dim

        self.proprio_encoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, int(self.z_dim)),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initialize_weights:
            init_weights(self.modules())

    def forward(self, proprio):
        """Apply ProprioEncoder to Proprio Input.

        Args:
            proprio (torch.Tensor): Proprio Input

        Returns:
            torch.Tensor: Encoded Output
        """
        return self.proprio_encoder(proprio.float())


class ForceEncoder(nn.Module):
    """Implements force encoder module.

    Sourced from selfsupervised code.
    """

    def __init__(self, z_dim, initialize_weights=True):
        """Initialize ForceEncoder Module.

        Args:
            z_dim (float): Z dimension size
            initialize_weights (bool, optional): Whether to initialize weights or not. Defaults to True.
        """
        super().__init__()
        self.z_dim = int(z_dim)

        self.frc_encoder = nn.Sequential(
            CausalConv1D(6, 16, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(16, 32, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(32, 64, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(64, 128, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(128, self.z_dim, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initialize_weights:
            init_weights(self.modules())

    def forward(self, force):
        """Apply ForceEncoder to Force Input.

        Args:
            force (torch.Tensor): Force Input, shape (*, 32, 6)

        Returns:
            torch.Tensor: Encoded Output
        """
        force = force.transpose(-1, -2).float() # shape (*, 6, 32)
        out = self.frc_encoder(force)
        return out.view(len(out), -1)


class ActionEncoder(nn.Module):
    """Implements an action-encoder module."""

    def __init__(self, action_dim):
        """Instantiate ActionEncoder module.

        Args:
            action_dim (int): Dimension of action.
        """
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, action):
        """Apply action encoder to action input.

        Args:
            action (torch.Tensor optional): Action input

        Returns:
            torch.Tensor: Encoded output
        """
        if action is None:
            return None
        return self.action_encoder(action)