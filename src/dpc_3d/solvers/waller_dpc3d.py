import numpy as np
import gc

# Try to import cupy; fall back to numpy if cupy is unavailable
try:
    import cupy as cp

    xp = cp
    gpu_available = True
    cp.fft._cache.PlanCache(memsize=0)
except ImportError:
    xp = np
    gpu_available = False

# Define Fourier Transform functions based on available package
F_2D = lambda x: xp.fft.fft2(x, axes=(0, 1))
IF_2D = lambda x: xp.fft.ifft2(x, axes=(0, 1))
F_3D = lambda x: xp.fft.fftn(x, axes=(0, 1, 2))
IF_3D = lambda x: xp.fft.ifftn(x, axes=(0, 1, 2))

pi = xp.pi
naxis = xp.newaxis


def pupilGen(fxlin, fylin, wavelength, na, na_in=0.0):
    pupil = xp.array(
        fxlin[naxis, :] ** 2 + fylin[:, naxis] ** 2 <= (na / wavelength) ** 2,
        dtype=xp.float32,
    )
    if na_in != 0.0:
        pupil[
            fxlin[naxis, :] ** 2 + fylin[:, naxis] ** 2 < (na_in / wavelength) ** 2
        ] = 0.0
    return pupil


def _genGrid(size, dx):
    xlin = xp.arange(size, dtype=xp.complex64)
    return (xlin - size // 2) * dx


def ensure_gpu_array(arr):
    """Ensure the array is on the GPU if using cupy."""
    if gpu_available and not isinstance(arr, cp.ndarray):
        return cp.asarray(arr)
    elif not gpu_available and isinstance(arr, cp.ndarray):
        return arr.get()
    return arr


class Solver3DDPC:
    def __init__(
        self,
        dpc_imgs,
        wavelength,
        na,
        na_in,
        pixel_size,
        pixel_size_z,
        rotation,
        RI_medium,
    ):
        self.wavelength = wavelength
        self.na = na
        self.na_in = na_in
        self.pixel_size = pixel_size
        self.pixel_size_z = pixel_size_z
        self.rotation = rotation
        self.dpc_num = len(rotation)
        self.fxlin = xp.fft.ifftshift(
            _genGrid(dpc_imgs.shape[1], 1.0 / dpc_imgs.shape[1] / self.pixel_size)
        )
        self.fylin = xp.fft.ifftshift(
            _genGrid(dpc_imgs.shape[0], 1.0 / dpc_imgs.shape[0] / self.pixel_size)
        )
        self.dpc_imgs = ensure_gpu_array(dpc_imgs.astype(xp.float32))
        self.RI_medium = RI_medium
        self.window = xp.fft.ifftshift(xp.hamming(dpc_imgs.shape[2]))
        self.pupil = pupilGen(self.fxlin, self.fylin, self.wavelength, self.na)
        self.phase_defocus = (
            self.pupil
            * 2.0
            * pi
            * xp.sqrt(
                (1.0 / wavelength) ** 2
                - self.fxlin[naxis, :] ** 2
                - self.fylin[:, naxis] ** 2
            )
        )
        self.oblique_factor = self.pupil / (
            4.0
            * pi
            * xp.sqrt(
                (RI_medium / wavelength) ** 2
                - self.fxlin[naxis, :] ** 2
                - self.fylin[:, naxis] ** 2
            )
        )
        self.sourceGen()
        self.WOTFGen()

    def sourceGen(self):
        self.source = []
        pupil = pupilGen(
            self.fxlin, self.fylin, self.wavelength, self.na, na_in=self.na_in
        )
        for rot_index in range(self.dpc_num):
            source = xp.zeros((self.dpc_imgs.shape[:2]), dtype=xp.float32)
            rotdegree = self.rotation[rot_index]
            condition = self.fylin[:, naxis] * xp.cos(
                xp.deg2rad(rotdegree)
            ) + 1e-15 >= self.fxlin[naxis, :] * xp.sin(xp.deg2rad(rotdegree))
            if rotdegree < 180:
                source[condition] = 1.0
            else:
                source[condition] = -1.0
                source += pupil
            source *= pupil
            self.source.append(source)
            del source, condition
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
        self.source = xp.array(self.source)

    def sourceFlip(self, source):
        source_flip = xp.fft.fftshift(source)[::-1, ::-1]
        if source_flip.shape[0] % 2 == 0:
            source_flip = xp.roll(source_flip, 1, axis=0)
        if source_flip.shape[1] % 2 == 0:
            source_flip = xp.roll(source_flip, 1, axis=1)
        return xp.fft.ifftshift(source_flip)

    def WOTFGen(self):
        dim_x = self.dpc_imgs.shape[1]
        dim_y = self.dpc_imgs.shape[0]
        dfx = 1.0 / dim_x / self.pixel_size
        dfy = 1.0 / dim_y / self.pixel_size
        z_lin = xp.fft.ifftshift(_genGrid(self.dpc_imgs.shape[2], self.pixel_size_z))
        prop_kernel = xp.exp(
            1.0j * z_lin[naxis, naxis, :] * self.phase_defocus[:, :, naxis]
        )
        self.H_real, self.H_imag = [], []
        for rot_index in range(self.dpc_num):
            source_flip = self.sourceFlip(self.source[rot_index])
            FSP_cFPG = (
                F_2D(source_flip[:, :, naxis] * self.pupil[:, :, naxis] * prop_kernel)
                * F_2D(
                    self.pupil[:, :, naxis]
                    * prop_kernel
                    * self.oblique_factor[:, :, naxis]
                ).conj()
            )
            H_real = (
                2.0
                * IF_2D(1.0j * FSP_cFPG.imag * dfx * dfy)
                * self.window[naxis, naxis, :]
            )
            H_real = xp.fft.fft(H_real, axis=2) * self.pixel_size_z
            H_imag = (
                2.0 * IF_2D(FSP_cFPG.real * dfx * dfy) * self.window[naxis, naxis, :]
            )
            H_imag = xp.fft.fft(H_imag, axis=2) * self.pixel_size_z
            total_source = (
                xp.sum(source_flip * self.pupil * self.pupil.conj()) * dfx * dfy
            )
            self.H_real.append(H_real * (1.0j / total_source))
            self.H_imag.append(H_imag / total_source)
            del FSP_cFPG, total_source, H_real, H_imag
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
        self.H_real = xp.array(self.H_real).astype(xp.complex64)
        self.H_imag = xp.array(self.H_imag).astype(xp.complex64)

    def _V2RI(self, V_real, V_imag):
        wavenumber = 2.0 * pi / self.wavelength
        B = -(self.RI_medium**2 - V_real / wavenumber**2)
        C = -((-V_imag / wavenumber**2 / 2.0) ** 2)
        RI_obj = xp.sqrt((-B + xp.sqrt(B**2 - 4.0 * C)) / 2.0)
        return RI_obj

    def setRegularizationParameters(
        self, reg_real=5e-5, reg_imag=5e-5, tau=5e-5, rho=5e-5
    ):
        self.reg_real = reg_real
        self.reg_imag = reg_imag
        self.tau = tau
        self.rho = rho

    def _prox_LASSO(self, V1_k, y_DV_k):
        DV_k_or_diff = xp.zeros(self.dpc_imgs.shape[:3] + (6,), dtype=xp.float32)
        DV_k_or_diff[..., 0] = V1_k[..., 0] - xp.roll(V1_k[..., 0], -1, axis=1)
        DV_k_or_diff[..., 1] = V1_k[..., 0] - xp.roll(V1_k[..., 0], -1, axis=0)
        DV_k_or_diff[..., 2] = V1_k[..., 0] - xp.roll(V1_k[..., 0], -1, axis=2)
        DV_k_or_diff[..., 3] = V1_k[..., 1] - xp.roll(V1_k[..., 1], -1, axis=1)
        DV_k_or_diff[..., 4] = V1_k[..., 1] - xp.roll(V1_k[..., 1], -1, axis=0)
        DV_k_or_diff[..., 5] = V1_k[..., 1] - xp.roll(V1_k[..., 1], -1, axis=2)

        DV_k = DV_k_or_diff - y_DV_k
        DV_k = xp.maximum(DV_k - self.tau / self.rho, 0.0) - xp.maximum(
            -DV_k - self.tau / self.rho, 0.0
        )
        DV_k_or_diff = DV_k - DV_k_or_diff
        return DV_k, DV_k_or_diff

    def _prox_projection(self, V1_k, V2_k, y_V2_k, boundary_constraint):
        V2_k = V1_k + y_V2_k
        V_real = V2_k[..., 1]
        V_imag = V2_k[..., 0]
        if boundary_constraint["real"] == "positive":
            V_real[V_real < 0.0] = 0.0
        elif boundary_constraint["real"] == "negative":
            V_real[V_real > 0.0] = 0.0
        if boundary_constraint["imag"] == "positive":
            V_imag[V_imag < 0.0] = 0.0
        elif boundary_constraint["imag"] == "negative":
            V_imag[V_imag > 0.0] = 0.0
        V2_k[..., 0] = V_imag
        V2_k[..., 1] = V_real
        return V2_k

    def _deconvTikhonov(self, AHA, AHy, determinant):
        V_real = IF_3D((AHA[0] * AHy[1] - AHA[2] * AHy[0]) / determinant).real
        V_imag = IF_3D((AHA[3] * AHy[0] - AHA[1] * AHy[1]) / determinant).real
        return V_real, V_imag

    def _deconvTV(
        self,
        AHA,
        determinant,
        fIntensity,
        fDx,
        fDy,
        fDz,
        tv_max_iter,
        boundary_constraint,
    ):
        AHy = [
            (self.H_imag.conj() * fIntensity).sum(axis=0),
            (self.H_real.conj() * fIntensity).sum(axis=0),
        ]
        V1_k = xp.zeros(self.dpc_imgs.shape[:3] + (2,), dtype=xp.float32)
        V2_k = xp.zeros(self.dpc_imgs.shape[:3] + (2,), dtype=xp.float32)
        DV_k = xp.zeros(self.dpc_imgs.shape[:3] + (6,), dtype=xp.float32)
        y_DV_k = xp.zeros(self.dpc_imgs.shape[:3] + (6,), dtype=xp.float32)
        y_V2_k = xp.zeros(self.dpc_imgs.shape[:3] + (2,), dtype=xp.float32)

        for iteration in range(tv_max_iter):
            AHy_k = [
                AHy[0]
                + self.rho
                * (
                    F_3D(V2_k[..., 0] - y_V2_k[..., 0])
                    + fDx.conj() * F_3D(DV_k[..., 0] + y_DV_k[..., 0])
                    + fDy.conj() * F_3D(DV_k[..., 1] + y_DV_k[..., 1])
                    + fDz.conj() * F_3D(DV_k[..., 2] + y_DV_k[..., 2])
                ),
                AHy[1]
                + self.rho
                * (
                    F_3D(V2_k[..., 1] - y_V2_k[..., 1])
                    + fDx.conj() * F_3D(DV_k[..., 3] + y_DV_k[..., 3])
                    + fDy.conj() * F_3D(DV_k[..., 4] + y_DV_k[..., 4])
                    + fDz.conj() * F_3D(DV_k[..., 5] + y_DV_k[..., 5])
                ),
            ]

            V1_k[..., 1], V1_k[..., 0] = self._deconvTikhonov(AHA, AHy_k, determinant)
            DV_k, DV_k_diff = self._prox_LASSO(V1_k, y_DV_k)
            V2_k = self._prox_projection(V1_k, V2_k, y_V2_k, boundary_constraint)
            y_DV_k += DV_k_diff
            y_V2_k += V1_k - V2_k
            

        return V1_k[..., 1], V1_k[..., 0]

    def solve(
        self,
        method="Tikhonov",
        tv_max_iter=20,
        boundary_constraint={"real": "negative", "imag": "negative"},
    ):
        AHA = [
            (self.H_imag.conj() * self.H_imag).sum(axis=0),
            (self.H_imag.conj() * self.H_real).sum(axis=0),
            (self.H_real.conj() * self.H_imag).sum(axis=0),
            (self.H_real.conj() * self.H_real).sum(axis=0),
        ]
        fIntensity = F_3D(self.dpc_imgs).transpose(3, 0, 1, 2).astype(xp.complex64)

        if method == "Tikhonov":
            AHA[0] += self.reg_imag
            AHA[3] += self.reg_real
            AHy = [
                (self.H_imag.conj() * fIntensity).sum(axis=0),
                (self.H_real.conj() * fIntensity).sum(axis=0),
            ]
            determinant = AHA[0] * AHA[3] - AHA[1] * AHA[2]
            V_real, V_imag = self._deconvTikhonov(AHA, AHy, determinant)
            
            del AHA, fIntensity, AHy, determinant
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        elif method == "TV":
            fDx = xp.zeros(self.dpc_imgs.shape[:3], dtype=xp.complex64)
            fDy = xp.zeros(self.dpc_imgs.shape[:3], dtype=xp.complex64)
            fDz = xp.zeros(self.dpc_imgs.shape[:3], dtype=xp.complex64)
            fDx[0, 0, 0] = 1.0
            fDx[0, -1, 0] = -1.0
            fDy[0, 0, 0] = 1.0
            fDy[-1, 0, 0] = -1.0
            fDz[0, 0, 0] = 1.0
            fDz[0, 0, -1] = -1.0
            fDx = F_3D(fDx).astype(xp.complex64)
            fDy = F_3D(fDy).astype(xp.complex64)
            fDz = F_3D(fDz).astype(xp.complex64)
            AHA[0] += self.rho * (
                fDx * fDx.conj() + fDy * fDy.conj() + fDz * fDz.conj() + 1.0
            )
            AHA[3] += self.rho * (
                fDx * fDx.conj() + fDy * fDy.conj() + fDz * fDz.conj() + 1.0
            )
            determinant = AHA[0] * AHA[3] - AHA[1] * AHA[2]
            V_real, V_imag = self._deconvTV(
                AHA,
                determinant,
                fIntensity,
                fDx,
                fDy,
                fDz,
                tv_max_iter,
                boundary_constraint,
            )

        RI_obj = self._V2RI(V_real, V_imag)
        del V_real, V_imag
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        return RI_obj
