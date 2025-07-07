import gc
import numpy as np

try:
    import cupy as cp # type: ignore
    xp = cp
    gpu_available = True
    cp.fft._cache.PlanCache(memsize=0)
except ImportError:
    xp = np
    gpu_available = False

# FFT shorthands
def F_2D(x):
    return xp.fft.fft2(x, axes=(0, 1))
def IF_2D(x):
    return xp.fft.ifft2(x, axes=(0, 1))
def F_3D(x):
    return xp.fft.fftn(x, axes=(0, 1, 2))
def IF_3D(x):
    return xp.fft.ifftn(x, axes=(0, 1, 2))

pi = xp.pi
naxis = xp.newaxis

def pupilGen(
    fxlin: np.ndarray,
    fylin: np.ndarray,
    wavelength: float,
    na: float,
    na_in: float = 0.0
) -> np.ndarray:
    """
    Circular/annular pupil in Fourier space.

    Parameters
    ----------
    fxlin, fylin : array_like
        1D frequency coords (x, y).
    wavelength : float
        Illumination wavelength.
    na : float
        Objective NA.
    na_in : float, optional
        Inner NA for annulus.

    Returns
    -------
    ndarray
        Float32 mask.
    """
    mask = (fxlin[naxis, :]**2 + fylin[:, naxis]**2
            <= (na / wavelength)**2)
    p = xp.array(mask, dtype=xp.float32)
    if na_in:
        inner = (fxlin[naxis, :]**2 + fylin[:, naxis]**2
                 < (na_in / wavelength)**2)
        p[inner] = 0.0
    return p


def _genGrid(size: int, dx: float) -> np.ndarray:
    """
    Centered 1D coordinate vector.

    Parameters
    ----------
    size : int
        Number of points.
    dx : float
        Step size.

    Returns
    -------
    complex64 ndarray
    """
    x = xp.arange(size, dtype=xp.complex64)
    return (x - size // 2) * dx


def ensure_gpu_array(arr):
    """
    If using CuPy, ensure `arr` is on GPU; otherwise move to host.
    """
    if gpu_available and not isinstance(arr, cp.ndarray):
        return cp.asarray(arr)
    if not gpu_available and isinstance(arr, cp.ndarray):
        return arr.get()
    return arr


class Solver3DDPC:
    """
    3D-DPC solver with chunked, in-place GPU memory usage.

    Parameters
    ----------
    dpc_imgs : array_like
        Stack shape (Py, Px, Pz, Ns).
    wavelength : float
        Illumination λ (μm).
    na : float
        Objective NA.
    na_in : float
        Inner NA.
    pix_size_xy : float
        Pixel pitch XY (μm).
    pix_size_z : float
        Axial step (μm).
    rotation : sequence of float
        Source angles.
    RI_med : float
        Background refractive index.
    """

    def __init__(
        self, dpc_imgs, wavelength, na, na_in,
        pix_size_xy, pix_size_z, rotation, RI_med
    ):
        self.wavelength = wavelength
        self.na = na
        self.na_in = na_in
        self.pix_xy = pix_size_xy
        self.pix_z = pix_size_z
        self.rotation = rotation
        self.dpc_num = len(rotation)

        Py, Px, Pz, Ns = dpc_imgs.shape
        self.fxlin = xp.fft.ifftshift(
            _genGrid(Px, 1.0 / Px / self.pix_xy)
        )
        self.fylin = xp.fft.ifftshift(
            _genGrid(Py, 1.0 / Py / self.pix_xy)
        )

        self.dpc_imgs = ensure_gpu_array(
            dpc_imgs.astype(xp.float32)
        )
        self.RI_med = RI_med

        self.window = xp.fft.ifftshift(xp.hamming(Pz))

        self.pupil = pupilGen(
            self.fxlin, self.fylin, wavelength, na
        )
        self.phase_defocus = (
            self.pupil * 2 * pi *
            xp.sqrt((1.0 / wavelength)**2
                    - self.fxlin[naxis, :]**2
                    - self.fylin[:, naxis]**2)
        )
        self.oblique_factor = self.pupil / (
            4 * pi * xp.sqrt((RI_med / wavelength)**2
                             - self.fxlin[naxis, :]**2
                             - self.fylin[:, naxis]**2)
        )

        self.sourceGen()
        self.WOTFGen()

        # In-place buffers
        self._fbuf = xp.zeros_like(
            self.dpc_imgs, dtype=xp.complex64
        )
        self._den = xp.zeros(
            self.dpc_imgs.shape[:3], dtype=xp.float32
        )

        # FFT plan
        if gpu_available:
            try:
                self.fft_plan = cp.fft.get_fft_plan(
                    self._fbuf, axes=(0, 1, 2)
                )
            except Exception:
                self.fft_plan = None
        else:
            self.fft_plan = None

    def sourceGen(self):
        """
        Build DPC source patterns per rotation angle.
        """
        pupilsrc = pupilGen(
            self.fxlin, self.fylin,
            self.wavelength, self.na, self.na_in
        )
        out = []
        for rot in self.rotation:
            src = xp.zeros(self.dpc_imgs.shape[:2],
                           dtype=xp.float32)
            ang = xp.deg2rad(rot)
            cond = (self.fylin[:, naxis] * xp.cos(ang)
                    + 1e-15
                    >= self.fxlin[naxis, :] * xp.sin(ang))
            if rot < 180:
                src[cond] = 1.0
            else:
                src[cond] = -1.0
                src += pupilsrc
            src *= pupilsrc
            out.append(src)
        self.source = xp.stack(out, axis=0)

    def sourceFlip(self, s):
        """
        Flip a source pattern for pupil‐plane convolution.
        """
        sf = xp.fft.fftshift(s)[::-1, ::-1]
        if sf.shape[0] % 2 == 0:
            sf = xp.roll(sf, 1, axis=0)
        if sf.shape[1] % 2 == 0:
            sf = xp.roll(sf, 1, axis=1)
        return xp.fft.ifftshift(sf)

    def WOTFGen(self):
        """
        Generate real and imaginary WOTFs for each source.
        """
        Py, Px, Pz, Ns = self.dpc_imgs.shape
        dfx = 1.0 / Px / self.pix_xy
        dfy = 1.0 / Py / self.pix_xy
        zlin = xp.fft.ifftshift(_genGrid(Pz, self.pix_z))
        prop = xp.exp(
            1j * zlin[naxis, naxis, :]
            * self.phase_defocus[:, :, naxis]
        )

        Hr, Hi = [], []
        for idx in range(self.dpc_num):
            sf = self.sourceFlip(self.source[idx])
            FSP = (
                F_2D(sf[:, :, naxis] * self.pupil[:, :, naxis] * prop)
                * F_2D(self.pupil[:, :, naxis]
                       * prop
                       * self.oblique_factor[:, :, naxis]).conj()
            )
            HR = 2.0 * IF_2D(1j * FSP.imag * dfx * dfy)
            HR = xp.fft.fft(HR * self.window[naxis, naxis, :],
                            axis=2) * self.pix_z

            HI = 2.0 * IF_2D(FSP.real * dfx * dfy)
            HI = xp.fft.fft(HI * self.window[naxis, naxis, :],
                            axis=2) * self.pix_z

            norm = xp.sum(sf * self.pupil * self.pupil.conj()
                         ) * dfx * dfy
            Hr.append(HR * (1j / norm))
            Hi.append(HI / norm)

        self.H_real = xp.stack(Hr, axis=0).astype(xp.complex64)
        self.H_imag = xp.stack(Hi, axis=0).astype(xp.complex64)

    def setRegularizationParameters(
        self, reg_real, reg_imag, tau, rho
    ):
        """
        Set solver regularization parameters.

        Parameters
        ----------
        reg_real : float
            L2 weight on real part.
        reg_imag : float
            L2 weight on imag part.
        tau : float
            TV weight.
        rho : float
            ADMM penalty.
        """
        self.reg_real = reg_real
        self.reg_imag = reg_imag
        self.tau = tau
        self.rho = rho

    def _V2RI(self, Vr, Vi):
        """
        Convert scattering potential to refractive index.

        Returns
        -------
        array_like
        """
        k0 = 2 * pi / self.wavelength
        B = -(self.RI_med**2 - Vr / k0**2)
        C = -((-Vi / k0**2 / 2.0)**2)
        return xp.sqrt((-B + xp.sqrt(B**2 - 4 * C)) / 2.0)

    def solve(
        self,
        method: str = "Tikhonov",
        tv_max_iter: int = 20,
        boundary_constraint: dict | None = None
    ):
        """
        Solve 3D DPC via Tikhonov or TV.

        Parameters
        ----------
        method : {'Tikhonov','TV'}
        tv_max_iter : int
        boundary_constraint : dict
        """
        if boundary_constraint is None:
            boundary_constraint = {
                "real": "negative",
                "imag": "negative"
            }

        # build AHA
        Hcr = self.H_real.conj()
        Hci = self.H_imag.conj()
        AHA = [
            (Hci * self.H_imag).sum(axis=0),
            (Hci * self.H_real).sum(axis=0),
            (Hcr * self.H_imag).sum(axis=0),
            (Hcr * self.H_real).sum(axis=0),
        ]

        # forward FFT in-place
        if gpu_available and self.fft_plan is not None:
            cp.fft.fftn(
                self.dpc_imgs,
                axes=(0, 1, 2),
                plan=self.fft_plan,
                out=self._fbuf
            )
        else:
            xp.fft.fftn(self.dpc_imgs,
                        axes=(0, 1, 2),
                        out=self._fbuf)

        Fint = self._fbuf.transpose(3, 0, 1, 2).astype(xp.complex64)

        if method == "Tikhonov":
            AHA[0] += self.reg_imag
            AHA[3] += self.reg_real
            AHy = [
                (Hci * Fint).sum(axis=0),
                (Hcr * Fint).sum(axis=0),
            ]
            det = AHA[0]*AHA[3] - AHA[1]*AHA[2]
            Vr, Vi = self._deconvTikhonov(AHA, AHy, det)

            del AHy, det, Fint
            if gpu_available:
                cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        else:  # TV
            shape3 = self.dpc_imgs.shape[:3]
            fDx = xp.zeros(shape3, dtype=xp.complex64)
            fDy = xp.zeros(shape3, dtype=xp.complex64)
            fDz = xp.zeros(shape3, dtype=xp.complex64)
            fDx[0, 0, 0] = 1 
            fDx[0, -1, 0] = -1
            fDy[0, 0, 0] = 1
            fDy[-1, 0, 0] = -1
            fDz[0, 0, 0] = 1
            fDz[0, 0, -1] = -1

            xp.fft.fftn(fDx, axes=(0,1,2), out=fDx)
            xp.fft.fftn(fDy, axes=(0,1,2), out=fDy)
            xp.fft.fftn(fDz, axes=(0,1,2), out=fDz)

            tvm = (fDx*fDx.conj() + fDy*fDy.conj() + fDz*fDz.conj() + 1.0)
            AHA[0] += self.rho * tvm
            AHA[3] += self.rho * tvm
            det = AHA[0]*AHA[3] - AHA[1]*AHA[2]

            Vr, Vi = self._deconvTV(
                AHA, det, Fint, fDx, fDy, fDz,
                tv_max_iter, boundary_constraint
            )

            del fDx, fDy, fDz, Fint, det
            if gpu_available:
                cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        RI = self._V2RI(Vr, Vi)
        del Vr, Vi
        if gpu_available:
            cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        return RI
