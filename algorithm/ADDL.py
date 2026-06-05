from utils.utils import *
from utils.FRSGM.condrefinenet_fourier import CondRefineNetDilated

class TDL_DRP:
    def __init__(self,
                 max_iter=80):

        import main
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.T_mask, self.T_ksdata, self.T_ksfull = main.mask, main.ksdata, main.ksfull
        self.T_Ker, self.T_Ker_Tra, self.T_sensitivity = main.Ker, main.Ker_Tra, main.sensitivityLi

        self.max_iter, self.save_path = max_iter, main.save_path

        self.scoreNet = CondRefineNetDilated(2, 2, 64).to(device)
        self.scoreNet.load_state_dict(torch.load(f'the path of checkpoint'))
        self.scoreNet.eval()
        self.gamma = 1.5
        self.lam = 0.0001
        self.score_sigma = torch.zeros(1, 1, 1, 1, device=device)

        self.Lip_C = (1 + main.Lip_C) ** 2
        self.PD3O_gamma = 1.999 / self.Lip_C
        self.PD3O_delta = 0.999 / self.PD3O_gamma

        self.Thr = 0
        self.start_time = time.time() - main.t_kernel - main.t_lip
        self.A = lambda x: VidHaarDec3S(IFFT2_3D_N((1 - self.T_mask) * x), self.level)
        self.At = lambda x: (1 - self.T_mask) * FFT2_3D_N(VidHaarRec3S(x, self.level))

    def process(self):
        y = self.T_ksdata.clone()
        c = VidHaarDec3S(IFFT2_3D_N(self.T_ksdata), self.level)
        s = torch.zeros_like(c)
        ref = sos(IFFT2_3D_N(self.T_ksfull))
        gamma_At_sk = 0

        pbar = tqdm(range(self.max_iter), desc=f'TDL-DRP', ncols=150)
        for iter in pbar:
            pbar.set_postfix(loss=f"{torch.norm(torch.abs(self.T_ksfull - y)):.3f}")

            u = (1 - self.T_mask) * y + self.T_ksdata
            grad_f = Kernel_Rec_ks_C_I_Pro(u, self.T_Ker, 1)
            grad_f = Kernel_Rec_ks_C_I_Pro(grad_f, self.T_Ker_Tra, 1)
            gamma_grad_f = self.PD3O_gamma * grad_f
            gamma_At_sk_uk_gamma_grad_f = gamma_At_sk - (2 * u - y - gamma_grad_f)
            t = s - self.PD3O_delta * self.A(gamma_At_sk_uk_gamma_grad_f) + self.PD3O_delta * c
            s[..., [1, 2, 3, 4, 6, 7, 8, 9]] = t[..., [1, 2, 3, 4, 6, 7, 8, 9]] - self.SGM(iter, y, t.clone())
            gamma_At_sk = self.PD3O_gamma * self.At(s)
            y = u - gamma_grad_f - gamma_At_sk

        res = sos(IFFT2_3D_N((1 - self.T_mask) * y + self.T_ksdata))
        useTime = time.time() - self.start_time
        return PSNR_SSIM_HaarPSI(ref, res, 'TDL-DRP', self.save_path, useTime)

    def SGM(self, iter, uk, xk_delta_z, step=1):
        if iter in range(0, self.max_iter - 20, 5):
            coil_image = DenoiseByDiffusion(self, IFFT2_3D_N((1 - self.T_mask) * uk + self.T_ksdata), step=step)
            coil_image = IFFT2_3D_N((1 - self.T_mask) * FFT2_3D_N(coil_image) + self.T_ksdata)
            y = torch.abs(VidHaarDec3S(coil_image))
            Thr = imfilter_symmetric_4D(y[..., [1, 2, 3, 4, 6, 7, 8, 9]])
            self.Thr = EnergyScaling_4D(1 / Thr, xk_delta_z[..., [1, 2, 3, 4, 6, 7, 8, 9]])
        elif iter in range(self.max_iter - 20, self.max_iter, 5):
            coil_image = IFFT2_3D_N((1 - self.T_mask) * uk + self.T_ksdata)
            y = torch.abs(VidHaarDec3S(coil_image))
            Thr = imfilter_symmetric_4D(y[..., [1, 2, 3, 4, 6, 7, 8, 9]])
            self.Thr = EnergyScaling_4D(1 / Thr, xk_delta_z[..., [1, 2, 3, 4, 6, 7, 8, 9]])

        xk_delta_z[..., [1, 2, 3, 4, 6, 7, 8, 9]] = wthresh(xk_delta_z[..., [1, 2, 3, 4, 6, 7, 8, 9]], self.Thr)
        return xk_delta_z[..., [1, 2, 3, 4, 6, 7, 8, 9]]
