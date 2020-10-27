import torch

def where_update(w, v, data):
    v = torch.tensor(v, dtype=data.dtype, device=data.device)
    return torch.where(w, v, data)

def torch_rgb_to_hls(images):
    """
    Transform a batch of RGB images into HLS format.
    
    Parameters
    ----------
    images : torch.Tensor (shape (N, 3, H, W))
        The images to transform.
        
    Returns
    -------
    hls_images : torch.Tensor (shape (N, 3, H, W))
        Instead of red, green and blue there is hue, luminance and saturation.
    """
    #BS, C, H, W
    vmin, imin = images.min(dim=1)
    vmax, imax = images.max(dim=1)
    c = vmax - vmin
    zeros = c==0
    pc = where_update(zeros, 1, c)
    
    #hue
    pim = images/pc[:,None]
    r, g, b = pim[:,0], pim[:,1], pim[:,2]
    ph = (imax==0)*(g-b) + (imax==1)*(b-r+2) + (imax==2)*(r-g+4)
    h = where_update(zeros, 0, ph)/6 % 1
    
    #luminance
    l = (vmax + vmin)/2
    
    #saturation
    pl = where_update(zeros, 1, l)
    ps = c / (1 - torch.abs(2*pl - 1))
    s = where_update(zeros, 0, ps)
    
    return torch.stack([h,l,s], dim=1)

def torch_hls_to_rgb(hls_images):
    """
    Transform a batch of RGB images into HLS format.
    
    Parameters
    ----------
    hls_images : torch.Tensor (shape (N, 3, H, W))
        The images, where the channels correspond to hue, luminance and saturation.
        
    Returns
    -------
    images : torch.Tensor (shape (N, 3, H, W))
        Instead of hue, luminance and saturation there is red, green and blue.
    """
    im = hls_images
    h, l, s = im[:,[0]], im[:,[1]], im[:,[2]]
    a = s*torch.min(l, 1-l)
    h12 = 12*h
    n = torch.tensor([[[0]], [[8]], [[4]]], dtype=h12.dtype, device=h12.device)
    k = (n+h12)%12
    one = torch.tensor(1, dtype=h12.dtype, device=h12.device)
    kmin = torch.min(torch.min(k-3, 9-k), one)
    v = torch.max(-one,  kmin)
    return (l-a*torch.max(-one,  kmin)).clamp(0, 1)