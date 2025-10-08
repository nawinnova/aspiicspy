import numpy as np
import matplotlib.pyplot as plt
import sunpy.map
import sunkit_image.radial as radial
from sunkit_image.utils import equally_spaced_bins
import sunkit_image.enhance as enhance
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import AsymmetricPercentileInterval, ImageNormalize, LinearStretch, SqrtStretch
import glob
import os
from aspiicspy.generate_colormap import generate_colormap

def cut_occulter_sunpy(map, r=1.17):
    suncenter_pix = [map.meta['x_io'] -1, map.meta['y_io']-1]# Use IO center rather than sun center
    rsun_pix = map.meta['rsun_obs']/map.meta['CDELT1']
    x = np.arange(0, map.data.shape[0], 1)
    y = np.arange(0, map.data.shape[1], 1)
    xx, yy = np.meshgrid(x, y, sparse=True)
    dist_suncen = np.sqrt((xx - suncenter_pix[0])**2 + (yy - suncenter_pix[1])**2)
    map.data[dist_suncen < r*rsun_pix] = np.nan #nan pixel inside occulter

def read_aspiics_sunpy(filename, filter, exptime, occulter=True, rotate=True, inf_value= 'max', enhance_method=False):
    with fits.open(filename, do_not_scale_image_data=True) as hdul:             
       imagedata = hdul[0].data
       header    = hdul[0].header

    if header['filter'] != filter or np.round(header['exptime'], decimals=2) != np.float64(exptime):
        pass
    else:
        if inf_value == 'max':
            inf_value = np.max(imagedata[np.isfinite(imagedata)])
        imagedata[~np.isfinite(imagedata)] = inf_value
        fits_data = imagedata
        if enhance_method == 'WOW':
            denoise_coefficients = [5,2,1]
            gamma = 4
            fits_data = enhance.wow(fits_data, bilateral=1, denoise_coefficients=denoise_coefficients, gamma=gamma, h=0) 
        if enhance_method == 'MGN':
            fits_data = enhance.mgn(fits_data) # need to explore further
    
        image_sunpy = sunpy.map.Map(fits_data, header) # register to sunpy to handle coordinate stuff
        if occulter == True:
            cut_occulter_sunpy(image_sunpy, r=1.17) # use auxillary function to remove values from inside occulter (default r=1.17 rsun at 0% vignetting), as well as bad pixel
            # see SOC doc for vignetting
        if rotate == True:
            image_sunpy = image_sunpy.rotate() # rotate to solar north = up

        if enhance_method == 'NRGF':
            FOV_low = 1.17
            FOV_high = 3
            radial_bin_edges = equally_spaced_bins(FOV_low, FOV_high, nbins=100)*u.R_sun
            image_sunpy = radial.nrgf(image_sunpy[0], radial_bin_edges=radial_bin_edges, width_function = np.nanstd)

        if enhance_method != False:
            image_sunpy.plot_settings['norm'] = ImageNormalize(image_sunpy.data, stretch=LinearStretch(), interval = AsymmetricPercentileInterval(1, 99.9))
        else:
            image_sunpy.plot_settings['norm'] = ImageNormalize(image_sunpy.data, stretch=SqrtStretch(), interval = AsymmetricPercentileInterval(1, 99.9))

        return image_sunpy
    
def plot_image(aspiics_map, bottom_left = 'default', top_right = 'default',image_dir=None, return_map = False):
    if 'Polarizer' in aspiics_map.meta['filter']:
        cmap = 'ASPIICS pB'
    else:
        cmap = 'ASPIICS '+aspiics_map.meta['filter']
    
    exptime = np.round(aspiics_map.meta['exptime'], decimals=2)

    if bottom_left == 'default':
        bl = SkyCoord(-3000*u.arcsec, -3000*u.arcsec, frame=aspiics_map.coordinate_frame)
    else:
        bl = bottom_left
    if top_right == 'default':
        tr = SkyCoord(3000*u.arcsec, 3000*u.arcsec, frame=aspiics_map.coordinate_frame)
    else:
        tr = top_right
    
    aspiics_map = aspiics_map.submap(bl, top_right=tr)

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
    generate_colormap()
    datafolder = '/Users/ngampoopun/Desktop/ASPIICS_stuff/WB_fits_file/exp1s/'
    filename_list = sorted(glob.glob(datafolder+'*_l2_*.fits')) # only select l2 data!!
    filter = 'Wideband'
    exptime = 1
    image_dir = '/Users/ngampoopun/Desktop/ASPIICS_stuff/Plots/Jet_WB_1s/'
    enhance_method = False

    for filename in filename_list:
        print('Plotting ASPIICS map:', filename)
        aspiics_map = read_aspiics_sunpy(filename, filter, exptime, enhance_method=enhance_method)
        plot_image(aspiics_map, image_dir)
    

 