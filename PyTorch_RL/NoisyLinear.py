import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class NoisyNetworkMixin:
    """Mixin to recursively reset all Noisy layers within a module."""
    def reset_noise(self):
        with th.no_grad():
            for module in self.modules():
                # Any module that implements reset_noise() via the mixin
                if isinstance(module, (NoisyLinear, NoisySequential, NoisyNetworkMixin)):
                    # Avoid infinite recursion on self
                    if module is not self:
                        module.reset_noise()




class NoisyLinear(nn.Module):
    """
    Factorized Gaussian Noisy Linear layer for Rainbow DQN (Fortunato et al., 2017).

    Args:
        in_features (int): input size
        out_features (int): output size
        sigma_init (float): initial standard deviation for noise parameters
        dtype: optional torch dtype
        device: optional device

    Methods:
        reset_parameters(): initialize mu and sigma
        reset_noise(): sample new factorized Gaussian noise
        forward(x, deterministic=False): forward pass, noisy or deterministic
    """
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5, dtype=None, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = float(sigma_init)

        # --- Trainable parameters ---
        self.weight_mu = nn.Parameter(th.empty(out_features, in_features, dtype=dtype, device=device))
        self.weight_sigma = nn.Parameter(th.empty(out_features, in_features, dtype=dtype, device=device))
        self.bias_mu = nn.Parameter(th.empty(out_features, dtype=dtype, device=device))
        self.bias_sigma = nn.Parameter(th.empty(out_features, dtype=dtype, device=device))

        # --- Non-trainable noise buffers ---
        self.register_buffer('weight_epsilon', th.zeros(out_features, in_features, dtype=dtype, device=device))
        self.register_buffer('bias_epsilon', th.zeros(out_features, dtype=dtype, device=device))

        # Initialize parameters and sample initial noise
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize mu and sigma"""
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _f(x: th.Tensor):
        """Factorized noise transform: f(x) = sign(x) * sqrt(|x|)"""
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        """Sample new factorized Gaussian noise"""
        if th.compiler.is_compiling():
            return  # Skip resampling during torch.compile tracing
        with th.no_grad():
            # Sample noise for input and output dimensions
            eps_in = self._f(th.randn(self.in_features, device=self.weight_mu.device, dtype=self.weight_mu.dtype))
            eps_out = self._f(th.randn(self.out_features, device=self.weight_mu.device, dtype=self.weight_mu.dtype))
            
            # Outer product for weight noise
            self.weight_epsilon.copy_(eps_out.outer(eps_in))  # More explicit than ger()
            self.bias_epsilon.copy_(eps_out)

    def forward(self, x: th.Tensor, deterministic: bool = False):
        """
        Forward pass.

        Args:
            x: input tensor of shape [batch, in_features]
            deterministic: if True, ignore noise and use mu only

        Returns:
            output tensor of shape [batch, out_features]
        """
        if self.training and not deterministic:
            weight = self.weight_mu
            bias = self.bias_mu
        else:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(x, weight, bias)
    
    def extra_repr(self) -> str:
        """Extra representation string"""
        return f'in_features={self.in_features}, out_features={self.out_features}, sigma_init={self.sigma_init}'

class NoisySequential(nn.Sequential, NoisyNetworkMixin):
    def forward(self, x, deterministic=False):
        for layer in self:
            if isinstance(layer, (NoisyLinear, NoisySequential)):
                x = layer(x, deterministic=deterministic)
            else:
                x = layer(x)
        return x
