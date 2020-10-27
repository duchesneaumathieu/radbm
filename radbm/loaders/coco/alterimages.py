import torch
import numpy as np
from pyperlin import FractalPerlin2D
from torch.distributions.normal import Normal
from radbm.utils.torch import torch_rgb_to_hls, torch_hls_to_rgb

def squeeze_values(mu, x):
    norm1 = Normal(mu, 1)
    norm2 = Normal(-mu, 1)
    return norm2.cdf(norm1.icdf(x.clamp(0,1)))

def build_grid(N, H, W, device='cpu'):
    hlin = torch.linspace(-1, 1, H, device=device)
    wlin = torch.linspace(-1, 1, W, device=device)
    y, x = torch.meshgrid(hlin, wlin)
    return torch.stack([x, y], dim=2).expand(N, H, W, 2)

def rotation_matrices(theta):
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    rotation = torch.zeros(*theta.shape, 2, 2, dtype=theta.dtype, device=theta.device)
    rotation[:,0,0] = cos
    rotation[:,0,1] = -sin
    rotation[:,1,0] = sin
    rotation[:,1,1] = cos
    return rotation

def spatial_batch_rotation(rotation, grid):
    #rotation.shape = (N, 2, 2)
    #grid.shape = (N, H, W, 2)
    return torch.matmul(rotation[:,None,None], grid[:,:,:,:,None])[:,:,:,:,0]

class AlterImages(torch.nn.Module):
    def __init__(
        self,
        shape,
        batch_global_scale_range = [1., 1.],
        hls_perlin_resolutions = [(2**i,2**i) for i in range(1,8)],
        hls_perlin_factors = [.5**i for i in range(7)],
        hls_global_noise_range = [[-.1, -.2, -.2], [.1, .2, .2]],
        hls_perlin_scale = [.5, 1, 1],
        spatial_distortion_perlin_resolutions = [(2**i,2**i) for i in range(1,4)],
        spatial_distortion_perlin_factors = [.5**i for i in range(3)],
        spatial_distortion_scale = .2,
        spatial_rotation_range = [-np.pi/16, np.pi/16],
        generator = torch.random.default_generator,
    ):
        self.shape = shape
        self.hls_perlin = FractalPerlin2D(
            shape,
            hls_perlin_resolutions,
            hls_perlin_factors,
            generator = generator
        )
        self.spatial_distortion_perlin = FractalPerlin2D(
            shape,
            spatial_distortion_perlin_resolutions,
            spatial_distortion_perlin_factors,
            generator = generator
        )
        self.device = generator.device
        self.generator = generator
        self.batch_global_scale_range = batch_global_scale_range
        self.hls_global_noise_range = torch.tensor(
            hls_global_noise_range, dtype=torch.float32, device=self.device)
        self.hls_perlin_scale = torch.tensor(
            hls_perlin_scale, dtype=torch.float32, device=self.device)[None, :, None, None]
        self.spatial_distortion_scale = spatial_distortion_scale
        self.spatial_rotation_range = spatial_rotation_range
        
    def uniform_noise(self, start, end, shape, dtype=torch.float32, device=None):
        device = self.device if device is None else device
        noise = torch.zeros(shape, dtype=dtype, device=device)
        noise.uniform_(start, end, generator=self.generator)
        return noise
    
    def alter_hls(self, images, batch_global_scale=None):
        N = len(images)
        perlin_noise = self.hls_perlin(batch_size=N*3)
        perlin_noise = perlin_noise.view(N, 3, *self.shape)#.permute(0,2,3,1)
        global_noise = self.uniform_noise(0, 1, images.shape[:2])
        global_range = self.hls_global_noise_range[1] - self.hls_global_noise_range[0]
        global_noise = global_range*global_noise + self.hls_global_noise_range[0]
        noise = self.hls_perlin_scale*perlin_noise + global_noise[:,:,None, None]
        if batch_global_scale is not None:
            noise = batch_global_scale[:,None,None,None]*noise
        images = torch_rgb_to_hls(images)
        h = (images[:,[0]] + noise[:,[0]]) % 1
        ls = squeeze_values(noise[:,1:], images[:,1:])
        images = torch.cat([h, ls], dim=1)
        return torch_hls_to_rgb(images)
    
    def alter_space(self, images, batch_global_scale=None):
        N = len(images)
        perlin_noise = self.spatial_distortion_perlin(batch_size=N*2)
        perlin_noise = perlin_noise.view(N, 2, *self.shape).permute(0,2,3,1)
        grid = build_grid(N, *self.shape, device=self.device)
        theta = self.uniform_noise(*self.spatial_rotation_range, len(images))
        if batch_global_scale is not None:
            theta = batch_global_scale*theta
            perlin_noise = batch_global_scale[:,None,None,None]*perlin_noise
        grid = spatial_batch_rotation(rotation_matrices(theta), grid)
        grid = grid + self.spatial_distortion_scale*perlin_noise
        return torch.nn.functional.grid_sample(images, grid, align_corners=False, padding_mode='zeros')
    
    def __call__(self, images):
        batch_global_scale = self.uniform_noise(*self.batch_global_scale_range, len(images))
        images = self.alter_hls(images, batch_global_scale=batch_global_scale)
        images = self.alter_space(images, batch_global_scale=batch_global_scale)
        return images