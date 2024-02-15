from __future__ import annotations

import typing as t
from dataclasses import dataclass

import mcubes
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    """Stores Gaussians

    Args:
        mean: Mean of multivariate Gaussian
        cov: Covariance of multivariate Gaussian.
    """

    mean: Float[Tensor, "*batch dim"]
    cov: Float[Tensor, "*batch dim dim"]


def compute_3d_gaussian(
        directions: Float[Tensor, "*batch 3"],
        means: Float[Tensor, "*batch 3"],
        dir_variance: Float[Tensor, "*batch 1"],
        radius_variance: Float[Tensor, "*batch 1"],
) -> Gaussians:
    """Compute gaussian along ray.

    Args:
        directions: Axis of Gaussian.
        means: Mean of Gaussian.
        dir_variance: Variance along direction axis.
        radius_variance: Variance tangent to direction axis.

    Returns:
        Gaussians: Oriented 3D gaussian.
    """

    dir_outer_product = directions[..., :, None] * directions[..., None, :]
    eye = torch.eye(directions.shape[-1], device=directions.device)
    dir_mag_sq = torch.clamp(torch.sum(directions ** 2, dim=-1, keepdim=True), min=1e-10)
    null_outer_product = eye - directions[..., :, None] * (directions / dir_mag_sq)[..., None, :]
    dir_cov_diag = dir_variance[..., None] * dir_outer_product[..., :, :]
    radius_cov_diag = radius_variance[..., None] * null_outer_product[..., :, :]
    cov = dir_cov_diag + radius_cov_diag
    return Gaussians(mean=means, cov=cov)


def conical_frustum_to_gaussian(
        origins: Float[Tensor, "*batch 3"],
        directions: Float[Tensor, "*batch 3"],
        starts: Float[Tensor, "*batch 1"],
        ends: Float[Tensor, "*batch 1"],
        radius: Float[Tensor, "*batch 1"],
) -> Gaussians:
    """Approximates conical frustums with a Gaussian distributions.

    Uses stable parameterization described in mip-NeRF publication.

    Args:
        origins: Origins of cones.
        directions: Direction (axis) of frustums.
        starts: Start of conical frustums.
        ends: End of conical frustums.
        radius: Radii of cone a distance of 1 from the origin.

    Returns:
        Gaussians: Approximation of conical frustums
    """
    mu = (starts + ends) / 2.0
    hw = (ends - starts) / 2.0
    means = origins + directions * (mu + (2.0 * mu * hw ** 2.0) / (3.0 * mu ** 2.0 + hw ** 2.0))
    dir_variance = (hw ** 2) / 3 - (4 / 15) * ((hw ** 4 * (12 * mu ** 2 - hw ** 2)) / (3 * mu ** 2 + hw ** 2) ** 2)
    radius_variance = radius ** 2 * ((mu ** 2) / 4 + (5 / 12) * hw ** 2 - 4 / 15 * (hw ** 4) / (3 * mu ** 2 + hw ** 2))
    return compute_3d_gaussian(directions, means, dir_variance, radius_variance)


def get_gaussian_blob(origins, directions, starts, ends, pixel_area) -> Gaussians:
    """Calculates guassian approximation of conical frustum.

    Returns:
        Conical frustums approximated by gaussian distribution.
    """
    # Cone radius is set such that the square pixel_area matches the cone area.
    cone_radius = torch.sqrt(pixel_area) / 1.7724538509055159  # r = sqrt(pixel_area / pi)
    return conical_frustum_to_gaussian(origins=origins, directions=directions, starts=starts,
                                       ends=ends, radius=cone_radius, )


@torch.no_grad()
def extract_fields(
        bound_min: t.Tuple[float, float, float],
        bound_max: t.Tuple[float, float, float],
        resolution: int,
        query_func: t.Callable[[Tensor], Tensor]
):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)
                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


@torch.no_grad()
def sample_pdf(bins, weights, n_samples, det: bool = False):
    # This implementation is from NeRF
    # Get pdf
    weights += 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # make it size be N instead of N - 1
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denominator = (cdf_g[..., 1] - cdf_g[..., 0])
    denominator = torch.where(denominator < 1e-5, torch.ones_like(denominator), denominator)
    t = (u - cdf_g[..., 0]) / denominator
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples.detach()


def get_pts(rays_o: Tensor, rays_d: Tensor, z_vals: Tensor):
    return rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]


class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = get_pts(rays_o, rays_d, mid_z_vals)
        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)  # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        sampled_color = torch.sigmoid(sampled_color)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    @torch.no_grad()
    def _up_sample(self, *, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        args
            inv_s: value related to the temperature of the softmax.
        """
        batch_size, n_samples = z_vals.shape
        pts = get_pts(rays_o, rays_d, z_vals)  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_estimate_sdf = mid_sdf - cos_val * dist * 0.5
        next_estimate_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_estimate_sdf * inv_s)
        next_cdf = torch.sigmoid(next_estimate_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True)
        return z_samples

    @torch.no_grad()
    def _cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        """
        adding extra new_z_vals and  new_sdf to sfd and z_values
        """
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = get_pts(rays_o, rays_d, new_z_vals)
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,
                    *,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = get_pts(rays_o, rays_d, mid_z_vals)
        # from models.utils import visualize_rays
        # visualize_rays(pts[:20].reshape(-1, 3))
        # here is the code to integrate the gaussian blob
        #
        wide_shape = pts.shape

        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        gaussian_blob = get_gaussian_blob(origins=rays_o[:, None, :].expand(wide_shape),
                                          directions=rays_d[:, None, :].expand(wide_shape), starts=z_vals.unsqueeze(-1),
                                          ends=torch.cat([z_vals[..., 1:],
                                                          torch.Tensor([sample_dist]).expand(dists[..., :1].shape)],
                                                         -1).unsqueeze(-1),
                                          pixel_area=torch.ones(tuple([*wide_shape[:-1], 1])))

        sdf_nn_output = sdf_network(pts, cov=gaussian_blob.cov.reshape(-1, 3, 3))
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network.gradient(pts).squeeze()
        sampled_color = color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        # Render with background
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] + \
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:  # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        return {
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere
        }

    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples  # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if self.n_importance > 0:
            n_samples, z_vals = self.resampling(rays_d, rays_o, z_vals, up_sample_steps=self.up_sample_steps,
                                                n_importance=self.n_importance)

        # Background model
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # Render core
        ret_fine = self.render_core(rays_o=rays_o,
                                    rays_d=rays_d,
                                    z_vals=z_vals,
                                    sample_dist=sample_dist,
                                    sdf_network=self.sdf_network,
                                    deviation_network=self.deviation_network,
                                    color_network=self.color_network,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio)

        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        return {
            'color_fine': color_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere']
        }

    @torch.no_grad()
    def resampling(self, rays_d, rays_o, z_vals, up_sample_steps: int, n_importance: int):
        batch_size = rays_d.shape[0]
        pts = get_pts(rays_o, rays_d, z_vals)
        sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)

        assert n_importance % up_sample_steps == 0, f"{n_importance} should be divisible by {up_sample_steps}"
        cur_num_importance = n_importance // up_sample_steps
        for i in range(1, up_sample_steps + 1):
            new_z_vals = self._up_sample(rays_o=rays_o,
                                         rays_d=rays_d,
                                         z_vals=z_vals,
                                         sdf=sdf,
                                         n_importance=cur_num_importance,
                                         inv_s=64 * 2 ** i)
            z_vals, sdf = self._cat_z_vals(rays_o,
                                           rays_d,
                                           z_vals,
                                           new_z_vals,
                                           sdf,
                                           last=(i == up_sample_steps))
        n_samples = self.n_samples + n_importance
        return n_samples, z_vals

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))
