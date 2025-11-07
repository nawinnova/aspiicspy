import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sunpy.map
import sunkit_image.radial as radial
from sunkit_image.utils import equally_spaced_bins
import sunkit_image.enhance as enhance
from astropy.io import fits
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.visualization import AsymmetricPercentileInterval, ImageNormalize, LinearStretch, SqrtStretch
import glob
import os
import warnings
warnings.simplefilter('ignore')
# from aspiicspy.generate_colormap import generate_colormap

def query_condition(filename_input, filter=None, exptime=None, start_time=None, end_time=None):
    """
    Query the input ASPIICS filelist based on various conditions.

    Parameters:
        filename_input (list): List of input filenames.
        filter (str): Filter condition to apply. should be a string matching the 'filter' keyword in the FITS header.
            Options are: 'Wideband', 'Fe XIV', 'Polarizer 0', 'Polarizer 60', 'Polarizer 120', 'He I'
        exptime (float): Exposure time condition to apply.
        start_time (str): Start time for the query. Format: 'YYYY-MM-DDTHH:MM:SS'
        end_time (str): End time for the query. Format: 'YYYY-MM-DDTHH:MM:SS'

    Returns:
        list: Filtered list of filenames based on the specified conditions.
    """
    filename_list = filename_input[:]
    if start_time is None:
        start_time = Time('1970-01-01T00:00:00')
    else:
        start_time = Time(start_time)
    if end_time is None:
        end_time = Time('2100-01-01T00:00:00')
    else:
        end_time = Time(end_time)
    # print(start_time, end_time)
    for i, file in enumerate(filename_list):
        header = fits.getheader(file, -1)
        # filetime = Time(header['date-obs'])
        # filefilter =  header['filter']
        # fileexp = np.round(header['exptime'], decimals=2)
        if Time(header['date-obs']) < start_time or Time(header['date-obs']) > end_time:
            filename_list[i] = None
        elif filter is not None and header['filter'] != filter:
            filename_list[i] = None 
        elif exptime is not None and np.round(header['exptime'], decimals=2) != np.float64(exptime):
            filename_list[i] = None
        del header
    
    filename_list = [f for f in filename_list if f is not None]
    
    return filename_list
            

def cut_occulter_sunpy(map, r=1.17):
    """
    Cut the occulter region from a SunPy map.

    Parameters:
        map (sunpy.map.Map): The SunPy map to process.
        r (float): The radius of the occulter in solar radii (default is 1.17 to correspond to 0% vignetting).

    Returns:
        None: The function modifies the input map in place by setting pixels inside the occulter to NaN.
    """
    suncenter_pix = [map.meta['x_io'] -1, map.meta['y_io']-1]# Use IO center rather than sun center
    rsun_pix = map.meta['rsun_obs']/map.meta['CDELT1']
    x = np.arange(0, map.data.shape[0], 1)
    y = np.arange(0, map.data.shape[1], 1)
    xx, yy = np.meshgrid(x, y, sparse=True)
    dist_suncen = np.sqrt((xx - suncenter_pix[0])**2 + (yy - suncenter_pix[1])**2)
    map.data[dist_suncen < r*rsun_pix] = np.nan #nan pixel inside occulter


def read_aspiics_sunpy(filename, occulter=True, rotate=True, inf_value = 'max', enhance_method=False, savedir=None):
    """
    Read ASPIICS Level 2 FITS file and return a SunPy map with optional processing.
    
    Parameters:
        filename (str): Filename to the ASPIICS Level 2 FITS file.
        occulter (bool): Whether to cut the occulter region (default is True).
        rotate (bool): Whether to rotate the image to solar north up (default is True).
        inf_value (str or float): Value to replace infinite/NaN pixels. If 'max', replaces with max finite value (default is 'max'). If a number is provided, uses that value.
        enhance_method (str or bool, optional): Image enhancement method. 
            Options are 'WOW', 'MGN', 'NRGF', or False for no enhancement (default is False).
        savedir (str or None, optional): Directory to save the processed FITS file. If None, does not save (default is None).

    Returns:
        sunpy.map.Map: Processed SunPy map of the ASPIICS image.
    """

    with fits.open(filename, do_not_scale_image_data=True) as hdul:             
       imagedata = np.array(hdul[-1].data, dtype="<f4")
       header    = hdul[-1].header  

    if inf_value == 'max':
        inf_value = np.max(imagedata[np.isfinite(imagedata)])
        # inf_value = np.nanmax(imagedata)
    
    imagedata[~np.isfinite(imagedata)] = inf_value
    # imagedata[np.isnan(imagedata)] = inf_value

    fits_data = imagedata
    if enhance_method == 'WOW':
        denoise_coefficients = [5,2,1]
        gamma = 4
        fits_data = enhance.wow(fits_data, bilateral=1, denoise_coefficients=denoise_coefficients, gamma=gamma, h=0) 
    if enhance_method == 'MGN':
        fits_data = enhance.mgn(fits_data) # need to explore further

    image_sunpy = sunpy.map.Map(fits_data, header) # register to sunpy to handle coordinate stuff
    
    if occulter == True:
        cut_occulter_sunpy(image_sunpy, r=1.17) # cut occulter region
    if rotate == True:
        image_sunpy = image_sunpy.rotate() # rotate to solar north = up

    if enhance_method == 'NRGF':
        FOV_low = 1.17
        FOV_high = 3
        radial_bin_edges = equally_spaced_bins(FOV_low, FOV_high, nbins=500)*u.R_sun
        image_sunpy = radial.nrgf(image_sunpy[0], radial_bin_edges=radial_bin_edges, width_function = np.nanstd)

    if enhance_method != False:
        image_sunpy.plot_settings['norm'] = ImageNormalize(image_sunpy.data, stretch=LinearStretch(), interval = AsymmetricPercentileInterval(1, 99.9))
    else:
        image_sunpy.plot_settings['norm'] = ImageNormalize(image_sunpy.data, stretch=SqrtStretch(), interval = AsymmetricPercentileInterval(1, 99.9))
    
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
        exptime = np.round(image_sunpy.meta['exptime'], decimals=2)
        if exptime >= 1:
            exptime = int(exptime)
        image_sunpy.save(savedir+f'{image_sunpy.meta["filename"].split(".")[0]}_{str(exptime).replace(".","")}s.fits')

    return image_sunpy
    
def plot_image(aspiics_map, bottom_left = None, top_right = None, image_dir = None, return_map = False):
    """
    Plot ASPIICS SunPy map with optional cropping and saving. 
    Note that to use ASPIICS colormaps, the colormap generation function (aspiicspy.generate_colormap) must be called prior to this function.

    Parameters:
        aspiics_map (sunpy.map.Map): SunPy map of ASPIICS image.
        bottom_left (SkyCoord or None): Bottom left coordinate for submap (default is None).
        top_right (SkyCoord or None): Top right coordinate for submap (default is None).
        image_dir (str or None): Directory to save the plotted image. If None, does not save (default is None).
        return_map (bool): Whether to return the processed map (default is False).
    Returns:
        sunpy.map.Map or None: Processed SunPy map if return_map is True, otherwise None.
    """

    if 'Polarizer' in aspiics_map.meta['filter']:
        cmap = 'ASPIICS pB'
    else:
        cmap = 'ASPIICS '+aspiics_map.meta['filter']
    
    exptime = np.round(aspiics_map.meta['exptime'], decimals=2)

    if bottom_left is not None and top_right is not None:
        bl = bottom_left
        tr = top_right
        aspiics_map = aspiics_map.submap(bl, top_right=tr)

    with mpl.rc_context({'font.size':14}):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection=aspiics_map)
        aspiics_map.plot(axes=ax, cmap=cmap)
        aspiics_map.draw_limb()
        ax.set_title(aspiics_map.latex_name+ f' Exposure = {exptime} s')
        # ax.axis('off')
        if image_dir is not None:
            os.makedirs(image_dir, exist_ok=True)
            if exptime >= 1:
                exptime = int(exptime)
            fig.savefig(image_dir+f'{aspiics_map.meta['filename'].split('.')[0]}_{str(exptime).replace('.','')}s.png', dpi=200, bbox_inches='tight')
            plt.close()
    
    if return_map == True:
        return aspiics_map

if __name__ == '__main__':
    print('running read_L2 as script')
    from generate_colormap import generate_colormap
    generate_colormap()
    # datafolder = '/Users/ngampoopun/Desktop/ASPIICS_stuff/WB_fits_file/exp1s/'
    datafolder = '/Users/ngampoopun/Desktop/ASPIICS_stuff/lvl2_1_jitter_remap_3/'
    filename_list = sorted(glob.glob(datafolder+'*_l2_*.fits')) # only select l2 data!!
    filter = 'Wideband'
    exptime = 1
    # image_dir = '/Users/ngampoopun/Desktop/ASPIICS_stuff/Plots/Jet_WB_1s/center_LED_JC_new_remap/'
    image_dir = '/Users/ngampoopun/Desktop/ASPIICS_stuff/Plots/Jet_WB_1s/fullmap_JCcorrect_remap/'
    enhance_method = 'WOW'

    filename_list = query_condition(filename_list, filter, exptime)

    for filename in filename_list:
        print('Plotting ASPIICS map:', filename)
        aspiics_map = read_aspiics_sunpy(filename, occulter=False, rotate=False, enhance_method=enhance_method)
        # print(f'CRVAL1:{aspiics_map.meta['CRVAL1']} , CRVAL2:{aspiics_map.meta['CRVAL2']}')
        ### No cropping just pure image
        bl, tr = None, None
        ### LED position
        # bl = SkyCoord(-500*u.arcsec, -500*u.arcsec, frame = aspiics_map.coordinate_frame)
        # tr = SkyCoord(500*u.arcsec, 500*u.arcsec, frame = aspiics_map.coordinate_frame)
        ##### Plume position
        # bl = SkyCoord(-500*u.arcsec, -1500*u.arcsec, frame = aspiics_map.coordinate_frame)
        # tr = SkyCoord(500*u.arcsec, -900*u.arcsec, frame = aspiics_map.coordinate_frame)
        plot_image(aspiics_map, bottom_left=bl, top_right=tr, image_dir=image_dir)
    

 