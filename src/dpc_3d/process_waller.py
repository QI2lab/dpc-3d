import numpy as np
from tifffile import imread, TiffWriter
from dpc_3d.solvers.waller_dpc3d import Solver3DDPC
from pathlib import Path
from ryomen import Slicer
from tqdm import tqdm
import gc
from numpy.typing import ArrayLike
import typer

try:
    import cupy as cp
    xp = cp
    CUPY_AVIALABLE = True
    from cupyx.scipy import ndimage
except Exception:
    xp = np
    CUPY_AVIALABLE = False
    from scipy import ndimage

def replace_hot_pixels(
    noise_map: ArrayLike, 
    data: ArrayLike, 
    threshold: float = 375.0
) -> ArrayLike:
    """Replace hot pixels with median values surrounding them.

    Parameters
    ----------
    noise_map: ArrayLike
        darkfield image collected at long exposure time to get hot pixels
    data: ArrayLike
        ND data [broadcast_dim,z,y,x]

    Returns
    -------
    data: ArrayLike
        hotpixel corrected data
    """

    data = xp.asarray(data, dtype=xp.float32)
    noise_map = xp.asarray(noise_map, dtype=xp.float32)

    # threshold darkfield_image to generate bad pixel matrix
    hot_pixels = xp.squeeze(xp.asarray(noise_map))
    hot_pixels[hot_pixels <= threshold] = 0
    hot_pixels[hot_pixels > threshold] = 1
    hot_pixels = hot_pixels.astype(xp.float32)
    inverted_hot_pixels = xp.ones_like(hot_pixels) - hot_pixels.copy()

    data = xp.asarray(data, dtype=xp.float32)
    for z_idx in range(data.shape[0]):
        median = ndimage.median_filter(data[z_idx, :, :], size=3)
        data[z_idx, :] = inverted_hot_pixels * data[z_idx, :] + hot_pixels * median
    
    data[ data < 0] = 0

    if CUPY_AVIALABLE:
        data = xp.asnumpy(data).astype(np.uint16)
        gc.collect()
        cp.clear_memo()
        cp._default_memory_pool.free_all_blocks()
    else:
        data = data.astype(np.uint16)

    return data

app = typer.Typer()
app.pretty_exceptions_enable = False

@app.command()
def dpc3d_GPU(
    input_path: Path,
    wavelength_um: float = .461,
    chunk_size: int = 768,
    output_path: Path = None
):
    """3D DPC weak scattering reconstruction adapted from Waller lab.

    Parameters
    ----------
    input_path: Path
        Path to MM tiff file
    wavelength_um: float, default = .461
        Mean LED wavelength in microns
    chunk_size: int, default = 768
        Integer yx chunk size. (Suggested values 768, 512, or 256).
    output_path: Path
        Path to write DPC reconstruction .ome.tiff
    """
    
    # Load data and ensure it is uint16 data type
    print("Loading file...")
    dpc_images = imread(input_path).astype(np.uint16)
    print("Finished.")

    if output_path is None:
        output_path = input_path.parents[0] / Path(str(input_path.stem).strip(".ome")+"_dpc.ome.tiff")

    # Check if this acquisition is a single Z plane and therefore 3D.
    # If 3D, pad the array so that it is 4D because the code expects 4D
    if dpc_images.ndim == 3:
        dpc_images = np.expand_dims(dpc_images,axis=0) # e.g this would go from (4,2048,2048) to (1,4,2048,2048)

    # # hotpixel correction
    # print("Correcting hot pixels...")
    dpc_images = dpc_images.transpose(1,0,2,3)
    
     # comment out the following when using denoised images
    print(dpc_images.shape)
    for dpc_idx, dpc_image in enumerate(dpc_images):
         dpc_images[dpc_idx,:] = replace_hot_pixels(
             noise_map = np.max(dpc_image,axis=(0)),
             data = dpc_image,
             threshold = np.max(dpc_image,axis=(0,1,2))*.999       #hot pixel check napari to determine value
         )
         
    dpc_images = dpc_images.astype(np.float32)
    # print("Finished.")
    

    # z-stack normalization
    # loop over z-stacks so that calculation fits in the GPU memory.
    print("Normalizing...")
    for dpc_idx, dpc_image in enumerate(dpc_images):
        cp_dpc_image = cp.asarray(dpc_image,dtype=cp.float32)
        mean_intensity = cp.mean(cp_dpc_image, axis=(0, 1, 2), keepdims=True)
        cp_dpc_image = (cp_dpc_image / mean_intensity) - 1.0
        dpc_images[dpc_idx,:] = cp.asnumpy(cp_dpc_image).astype(np.float32)
    dpc_images = dpc_images.transpose(1,0,2,3)
    print("Finished.")
    '''
    print("Writing output...")
    with TiffWriter(normalized_output_path, bigtiff=True) as tif:
        metadata={
            'axes': 'ZYX',
           
        }
        options = dict(
            compression='zlib',
            compressionargs={'level': 8},
            photometric='minisblack',
        )
        tif.write(
            dpc_images.transpose(1,0,2,3).astype(np.float32),
            **options,
            metadata=metadata
        )
    print("Finished.")
    return
    '''
    # clean up GPU memory
    del cp_dpc_image
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    # Get dimension sizes for later use
    nz, npos, ny, nx = dpc_images.shape

    # Flip data to match Waller lab code expected dimension order
    dpc_images = dpc_images.transpose(2,3,0,1)

    # experiment metadata
    # TO DO: read as much as possible from image metadata
    wavelength     =  wavelength_um     #in micron              #green = 0.517  #red = 0.626  #blue = 0.461
    mag            =   20.0             #total magnification of the imaging system
    na             =    0.8             #numerical aperture of the imaging system
    na_in          =      0             #inner numerical aperture of the source, 0: conventional DPC pattern, >0: annular pattern
    pixel_size_cam =    2.4             #pixel size of camera in micron
    pixel_size     = pixel_size_cam/mag #in micron
    pixel_size_z   =    .65             #in micron
    rotation       = [270, 90, 0, 180]  #degree T=270 B=90 L=0 R=180 OG[90,270,0,180]
    RI_medium      = 1.33               #background refractive index (air = 1.0, water = 1.33, oil = 1.515)    
    tau_real       = 1e-4               #L2 penalty weight on real part of scattering potential     orginal value= 1e-4 doesnt work for 1e-2
    tau_imag       = 1e-4               #L2 penalty weight on imaginary part of scattering potential    orginal value= 1e-4

    # move normalized DPC data to GPU
    cp_dpc_images = cp.asarray(dpc_images,dtype=cp.float32)

    # create empty array to hold refractive index object
    RI_obj = np.zeros((ny,nx,nz),dtype=np.float32)
    
    # define crop sizes for calculation. 
    # Getting these right are key to fitting the calculation on the GPU
    if chunk_size == 768:
        overlap_chunk = 196
    elif chunk_size == 512:
        overlap_chunk = 131
    elif chunk_size == 256:
        overlap_chunk = 64
    crop_size = (chunk_size, chunk_size, nz, npos)  #using 768, 768 going to try 512 & 256  Next try 512&128
    overlap = (overlap_chunk, overlap_chunk, 0, 0)          #using 196, 196 orginally &64

    # execute chunked RI reconstruction on the GPU
    print("Estimating RI...")
    slices = Slicer(cp_dpc_images, crop_size=crop_size, overlap=overlap)
    first_run = True
    for crop, source, destination in tqdm(slices,desc="chunks"):
        # Run reconstruction on a crop of the whole image
        if first_run:
            solver_3ddpc = Solver3DDPC(crop, wavelength, na, na_in, pixel_size, pixel_size_z, rotation, RI_medium)
            solver_3ddpc.setRegularizationParameters(reg_real=tau_real, reg_imag=tau_imag, tau = 1e-6, rho=1e-2)    #orginally tauy 1e-6
            first_run = False

        solver_3ddpc.dpc_imgs = crop       
        RI_obj[destination[:-1]] = cp.asnumpy(solver_3ddpc.solve(method="TV", tv_max_iter=40, boundary_constraint={"real":"negative", "imag":"negative"})[source[:-1]]).astype(np.float32)
    print("Finished.")
    # Write output as compressed ome-tiff.
    print("Writing output...")
    with TiffWriter(output_path, bigtiff=True) as tif:
        metadata={
            'axes': 'ZYX',
            'SignificantBits': 32,
            'PhysicalSizeX': np.round(pixel_size,3),
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': np.round(pixel_size,3),
            'PhysicalSizeYUnit': 'µm',
            'PhysicalSizeZ': pixel_size_z,
            'PhysicalSizeZUnit': 'µm'
        }
        options = dict(
            compression='zlib',
            compressionargs={'level': 8},
            photometric='minisblack',
            resolutionunit='CENTIMETER',
        )
        tif.write(
            RI_obj.transpose(2,0,1),
            resolution=(
                1e4 / np.round(pixel_size,3),
                1e4 / np.round(pixel_size,3)
            ),
            **options,
            metadata=metadata
        )
    print("Finished.")

    # clean up GPU memory
    del cp_dpc_images, RI_obj
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

def main():
    app()  
    
if __name__ == "__main__":
    main()