# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
import torch
from torch import Tensor


def expected_sin(x_means: Tensor, x_vars: Tensor) -> Tensor:
    """Computes the expected value of sin(y) where y ~ N(x_means, x_vars)

    Args:
        x_means: Mean values.
        x_vars: Variance of values.

    Returns:
        torch.Tensor: The expected value of sin.
    """
    return torch.exp(-0.5 * x_vars) * torch.sin(x_means)


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.embed_fns, self.out_dim = self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            self.freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                def embed_fn(x, p_fn=p_fn, freq=freq):
                    return p_fn(x * freq)

                embed_fns.append(embed_fn)
                out_dim += d

        return embed_fns, out_dim

    def embed(self, inputs, cov=None):
        if cov is None:
            return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        return self._call_with_cov(inputs, cov)

    def _call_with_cov(self, inputs, cov):
        scaled_inputs = inputs[..., None] * self.freq_bands  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        input_var = torch.diagonal(cov, dim1=-2, dim2=-1)[..., :, None] * self.freq_bands[None, :] ** 2
        input_var = input_var.reshape((*input_var.shape[:-2], -1))

        encoded_inputs = expected_sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1),
                                      torch.cat(2 * [input_var], dim=-1))
        return torch.cat([inputs, encoded_inputs], dim=-1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {'include_input': True, 'input_dims': input_dims, 'max_freq_log2': multires - 1,
                    'num_freqs': multires, 'log_sampling': True, 'periodic_fns': [torch.sin, torch.cos], }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, cov=None, eo=embedder_obj): return eo.embed(x, cov=cov)

    return embed, embedder_obj.out_dim
