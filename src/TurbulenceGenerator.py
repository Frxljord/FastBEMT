import torch

class TurbulenceGenerator:
    def __init__(self, Lx, Ly, Lz, nx, ny, nz, sigma, L_scale, device='cuda'):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.sigma = sigma
        self.L_scale = L_scale
        self.device = device

        self.kx = 2 * torch.pi * torch.fft.fftfreq(nx, d=Lx/nx, device=device)
        self.ky = 2 * torch.pi * torch.fft.fftfreq(ny, d=Ly/ny, device=device)
        self.kz = 2 * torch.pi * torch.fft.fftfreq(nz, d=Lz/nz, device=device)

    def generate_turbulence(self):
        KX, KY, KZ = torch.meshgrid(self.kx, self.ky, self.kz, indexing='ij')
        K = torch.sqrt(KX**2 + KY**2 + KZ**2)
        
        K_sq = K**2
        L_sq_K_sq = (self.L_scale**2) * K_sq
        E_k = (1.4528 * self.sigma**2 * self.L_scale * (L_sq_K_sq)**2) / (1 + L_sq_K_sq)**(17/6)
        
        psd = E_k / (4 * torch.pi * K_sq)
        psd[0, 0, 0] = 0
        
        noise = torch.randn((self.nx, self.ny, self.nz), dtype=torch.complex64, device=self.device)
        
        box_complex = torch.fft.ifftn(torch.sqrt(psd) * noise)
        box = box_complex.real
        
        box = (box - box.mean()) / box.std() * self.sigma
        return box.cpu().numpy()