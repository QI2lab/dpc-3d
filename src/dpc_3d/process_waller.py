# process_waller.py

import gc
from pathlib import Path

import numpy as np
from tifffile import imread, TiffWriter
import typer
from tqdm import tqdm
from ryomen import Slicer
#import matplotlib.pyplot as plt

try:
    import cupy as cp # type: ignore
    xp = cp
    CUPY_AVAILABLE = True
    from cupyx.scipy import ndimage # type: ignore
except ImportError:
    xp = np
    CUPY_AVAILABLE = False
    from scipy import ndimage

from dpc_3d.solvers.waller_dpc3d import Solver3DDPC

app = typer.Typer()
app.pretty_exceptions_enable = False


def replace_hot_pixels(
    noise_map: np.typing.ArrayLike,
    data: np.typing.ArrayLike,
    threshold: float = 375.0
) -> np.ndarray:
    """
    Replace hot pixels with median of their 3×3 neighborhood.

    Parameters
    ----------
    noise_map : ArrayLike
        Dark-frame image to identify hot pixels.
    data : ArrayLike
        Stack [z, y, x] to correct.
    threshold : float, optional
        Pixel values above which are “hot” (default 375.0).

    Returns
    -------
    np.ndarray
        Corrected array, dtype uint16.
    """
    arr = xp.asarray(data, dtype=xp.float32)
    nm = xp.asarray(noise_map, dtype=xp.float32)

    mask = (nm > threshold).astype(xp.float32)
    inv = 1.0 - mask

    for z in range(arr.shape[0]):
        med = ndimage.median_filter(arr[z], size=3)
        arr[z] = inv * arr[z] + mask * med

    arr[arr < 0] = 0

    if CUPY_AVAILABLE:
        out = cp.asnumpy(arr).astype(np.uint16)
        del arr, med, mask, inv
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        return out

    return arr.astype(np.uint16)
"""
Help from gpt to attempt to plot the illumination sources mixed with our code and wallers
"""
def plot_dpc_sources(solver: Solver3DDPC) -> np.ndarray:
    
    if CUPY_AVAILABLE:
        sources = cp.asnumpy(solver.source)
    else:
        sources = solver.source
    return np.fft.fftshift(sources,axes=(-2,-1)).astype(np.float32)



@app.command()
def dpc3d_GPU(
    input_path: Path,
    wavelength_um: float = 0.461,
    chunk_size: int = 768,
    output_path: Path | None = None
) -> None:
    """
    3D DPC reconstruction (Waller lab) with chunked GPU memory usage.

    Parameters
    ----------
    input_path : Path
        Path to input TIFF (4D or 3D stack).
    wavelength_um : float, default=0.461
        LED wavelength (μm).
    chunk_size : int, default=768
        Y×X chunk size for GPU.
    output_path : Path, optional
        Destination OME-TIFF; by default appends `_dpc.ome.tiff`.
    """
    # ——— Load & hot-pixel correct ———
    print("Loading file...")
    imgs = imread(input_path).astype(np.float32)
    print("Done.")

    if output_path is None:
        stem = input_path.stem.strip(".ome")
        output_path = input_path.parent / f"{stem}_dpc.ome.tiff"

    # make 4D if needed
    if imgs.ndim == 3:
        imgs = imgs[None, ...]

    # reorder → [npos, z, y, x]
    imgs = imgs.transpose(1, 0, 2, 3)

    print("Correcting hot pixels...")
    for i in range(imgs.shape[0]):
       stack = imgs[i]
       noise = np.max(stack, axis=(0,1))
       thr = np.max(stack) * 0.999
       imgs[i] = replace_hot_pixels(noise, stack, thr)
    del stack, noise
    print("Done.")

    # TO DO: add denoising here.

    # ——— Normalize each z-stack ———
    print("Normalizing...")
    for i in range(imgs.shape[0]):
       arr = xp.asarray(imgs[i], dtype=xp.float32)
       mean_int = arr.mean(axis=(0, 1, 2), keepdims=True)
       arr = (arr / mean_int) - 1.0
       if CUPY_AVAILABLE:
           imgs[i] = cp.asnumpy(arr).astype(np.float32)
       else:
           imgs[i] = arr
    print("Done.")
    
    # reorder → [y, x, z, npos]
    imgs = imgs.transpose(2, 3, 1, 0)
    ny, nx, nz, npos = imgs.shape

    # experiment parameters
    mag = 20.0
    na = 0.8
    na_in = 0.0
    pix_xy = 2.4 / mag
    pix_z = 0.666*0.65  #scale factor of our stage*micromanager's aquition z movement
    rotation       = [0, 180, 90, 270]
    RI_med = 1.33
    tau_r = 1e-4
    tau_i = 1e-4

    RI_obj = np.zeros((ny, nx, nz), dtype=np.float32)

    # determine overlap
    if chunk_size >= 768:
        ov = 196
    elif chunk_size == 512:
        ov = 131
    elif chunk_size == 256:
        ov = 64
    else:
        ov = chunk_size // 4

    crop_sz = (chunk_size, chunk_size, nz, npos)
    overlap = (ov, ov, 0, 0)

    print("Estimating RI...")
    slicer = Slicer(imgs, crop_size=crop_sz, overlap=overlap)

    first = True
    for crop, source, dest in tqdm(slicer, desc="chunks"):
        if first:
            solver = Solver3DDPC(
                cp.asarray(crop,dtype=cp.float32), 
                wavelength_um, 
                na, 
                na_in,
                pix_xy, 
                pix_z, 
                rotation, 
                RI_med
            )
            


            solver.setRegularizationParameters(
                reg_real=tau_r,
                reg_imag=tau_i,
                tau=1e-6,
                rho=1e-2
            )

            # Save the source patterns to a .tif stack
            source_stack = plot_dpc_sources(solver)
            # import napari
            # viewer = napari.Viewer()
            # viewer.add_image(source_stack)
            # napari.run()
            
            with TiffWriter(output_path.parent / "illumination_patterns_waller.tif") as tif:
                meta = {"axes": "IYX"}
                tif.write(source_stack, metadata= meta, photometric="minisblack")
            print("Saved source stack:", source_stack.shape, source_stack.dtype)

            first = False
        else:
            # copy real-space chunk
            solver.dpc_imgs = cp.asarray(crop,dtype=cp.float32)


        rec = solver.solve(
            method="Tikhonov",
            tv_max_iter=40,
            boundary_constraint={"real": "negative", "imag": "negative"}
        )

        # extract interior and write to RI_obj
        RI_obj[dest[:-1]] = cp.asnumpy(rec[source[:-1]])

        # free per-chunk GPU memory
        del solver.dpc_imgs, rec
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
    print("Done.")
    
    # write out
    print("Writing output...")
    with TiffWriter(output_path, bigtiff=True) as tif:
        meta = {
            "axes": "ZYX",
            "PhysicalSizeX": pix_xy,
            "PhysicalSizeY": pix_xy,
            "PhysicalSizeZ": pix_z,
        }
        opts = {
            "compression": "zlib",
            "compressionargs": {"level": 8},
            "photometric": "minisblack",
        }
        tif.write(RI_obj.transpose(2, 0, 1), **opts, metadata=meta)
    print("Done.")

    # cleanup
    del RI_obj
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()


def main() -> None:
    """CLI entry-point."""
    app()


if __name__ == "__main__":
    main()
