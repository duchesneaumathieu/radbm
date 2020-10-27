import unittest, colorsys, torch
from radbm.utils.torch import torch_hls_to_rgb, torch_rgb_to_hls

class TestTorchColor(unittest.TestCase):
    def test_torch_rgb_hls_convertions(self):
        torch.manual_seed(0xcafe)
        images = torch.rand(32,3,256,256)
        hls_images = torch_rgb_to_hls(images)
        some_pixels = [(1,2), (50,255), (42,42), (0,0)]
        for h, w in some_pixels:
            hls = torch.tensor(colorsys.rgb_to_hls(*images[0,:,h,w]))
            self.assertTrue(torch.allclose(hls_images[0,:,h,w], hls))
        rgb_images = torch_hls_to_rgb(hls_images)
        self.assertTrue(torch.allclose(images, rgb_images, atol=1e-6))