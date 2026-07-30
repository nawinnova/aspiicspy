import numpy as np
from astropy.io import fits
import os
import copy
import cv2

########## Code for fixing sun center position in the map header #########
def find_sun_center(fname):
    '''    
    Find the sun center position in the map header using the OPLF center derived from the header information.
    """
    Parameters
    ----------
    fname : str
        The filename of the ASPIICS L2 FITS file.

    Returns
    -------
    newcrpix_x : float
        Corrected reference pixel X coordinate (FITS standard, 1-based).
    newcrpix_y : float
        Corrected reference pixel Y coordinate (FITS standard, 1-based).

    Notes
    -----
    The function calculates the sun center position from the OPLF (Occulter Payload Frame) center. 
    The coordinates stored in the FITS header. The calculation uses:
    - OPLFOY, OPLFOZ: OPLF center distances from CPLF (Coronagraph Payload Frame) center
    - ISD: Inter-spacecraft distance
    - X_IO, Y_IO: IO center position (assumed to be CPLF center)
    - CDELT1: Pixel scale in arcsec/pixel

    Notes on CPLF coordinate -- Image coordinate (cf. Shestov et al. 2026):
    +z is +x in image frame
    +y is -y in image frame

    The function assumes only translational shift with no rotation.
    Only valid for exposure times < 5 seconds (LED positions reliable), and also for un-rotated images.
    """

    '''
    header = fits.getheader(fname)
    if header['CROTA'] == 0:
        print('Cannot use already rotated images (solar north = up), exiting')
        return None, None
    elif header['exptime'] > 5:
        print('Exposure time is greater than 5 seconds: LEDs positions are not reliable, exiting')
        return None, None

    x_io = header['X_IO'] # IO center of images in x-direction, assume to be CPLF centre
    y_io = header['Y_IO'] # IO center of images in y-direction, assume to be CPLF centre
    ISD = header['ISD']*1000 # work in mm unit to be consistent with SS code
    OPLFOY = header['OPLFOY'] # distance to OPLF center in y-direction in CPLF frame
    OPLFOZ = header['OPLFOZ'] # distance to OPLF center in z-direction in CPLF frame
    pixscale = header['CDELT1'] # pixel scale in arcsec/pixel

    oplfcen_z_pix = np.arctan(OPLFOZ/(ISD/1000.0))/np.pi*180.0*3600.0/pixscale
    oplfcen_y_pix = np.arctan(OPLFOY/(ISD/1000.0))/np.pi*180.0*3600.0/pixscale

    ## relation between image coord and CPLF coord
    # +z is +x in image frame
    # +y is -y in image frame
    ## assuming the centre of CPLF is at x_io, y_io
    cplf_x = x_io
    cplf_y = y_io

    newcrpix_x = oplfcen_z_pix + cplf_x 
    newcrpix_y = - oplfcen_y_pix + cplf_y 

    print('Approximate OPLF center from header (image pixels, FITS standard):')
    print(f"X: {newcrpix_x:8.4f}, Y: {newcrpix_y:8.4f}")

    # map_header_corr = copy.deepcopy(map.meta)
    # # print(map_header_corr['exptime'])
    # map_header_corr['CRPIX1'] = newcrpix_x # new reference pixel
    # map_header_corr['CRPIX2'] = newcrpix_y # new reference pixel
    # map_header_corr['CRVAL1'] = 0 # new reference arcsec value -> 0 = sun center
    # map_header_corr['CRVAL2'] = 0 # new reference arcsec value -> 0 = sun center

    return newcrpix_x, newcrpix_y

def fix_sun_center(filename, newcrpix_x, newcrpix_y, savepath=None):
    '''
    Fix the sun center position in the map header using the OPLF center derived from the header information.
    """
    Parameters
    ----------
    filename : str
        The filename of the ASPIICS L2 FITS file.
    newcrpix_x : float
        Corrected reference pixel X coordinate (FITS standard, 1-based).
    newcrpix_y : float
        Corrected reference pixel Y coordinate (FITS standard, 1-based).

    Returns
    -------
    None

    Notes
    -----
    The function updates the CRPIX1, CRPIX2, CRVAL1, and CRVAL2 keywords in the FITS header to reflect the new sun center position. 
    The corrected FITS file is saved with a modified filename indicating the correction.

    Only valid for un-rotated images.
    """
    '''

    # Open the FITS file and update the header
    # we do this way to avoid modifying the original file
    with fits.open(filename, do_not_scale_image_data=True) as hdul:             
       imagedata = hdul[0].data
       header    = hdul[0].header
    # update header
    header['CRPIX1'] = newcrpix_x
    header['CRPIX2'] = newcrpix_y
    header['CRVAL1'] = 0
    header['CRVAL2'] = 0

    # Save the corrected FITS file with a new name
    if savepath is None:
        new_filename = filename.replace('.fits', '_LED.fits')
    else:
        # os.makedirs(os.path.dirname(savepath), exist_ok=True)
        new_filename_base = os.path.basename(filename).replace('.fits', '_LED.fits')
        new_filename = os.path.join(os.path.dirname(savepath), new_filename_base)

    fits.writeto(new_filename, imagedata, header=header, overwrite=True)
    print(f"Saved sun-center corrected FITS file: {new_filename}")

    return None

def fix_sun_center_batch(filelist, savepath=None):
    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
    fname = [os.path.basename(f) for f in filelist]
    # get acquisition number from filename
    acq = []
    for f in fname:
        if f.split('_')[-1] == 'v3.fits':
            acq.append(f.split('_')[4][:-1])
            print(f"Acquisition number (v3): {f.split('_')[4][:-1]}")
        else:
            acq.append(f.split('_')[3][:-1])
            print(f"Acquisition number (v2): {f.split('_')[3][:-1]}")
    # group file with same acquisition number together.
    maps_same_acq = {}
    for i, acq in enumerate(acq):
        if acq not in maps_same_acq:
            maps_same_acq[acq] = []
        maps_same_acq[acq].append(filelist[i])

    for j in range(len(maps_same_acq)):
        flist_sameacq = sorted(maps_same_acq[list(maps_same_acq.keys())[j]])
        # find sun center correction , using the first file on the list, which corresponds to lowest exposure time
        print(f"Finding sun center from file: {flist_sameacq[0]}")
        suncen_x, suncen_y = find_sun_center(flist_sameacq[0])
        if suncen_x is None or suncen_y is None:
            print('Sun center could not be determined, skipping this acquisition.')
            print('------------------------------------------')
            continue
        if np.abs(suncen_x) > 1500 or np.abs(suncen_y) > 1500:
            print(f'Sun center is out of bounds, Try longer exposure image.: {flist_sameacq[1]}')
            suncen_x, suncen_y = find_sun_center(flist_sameacq[1])
        # apply the same correction to all files with the same acquisition number
        for f in flist_sameacq:
            print(f"Applying sun center correction to file: {f}")
            fix_sun_center(f, suncen_x, suncen_y, savepath=savepath)
        print('------------------------------------------')
        

######### Code for combining images with different exposure times #########
def find_multiplier(short_exptime, long_exptime, fit_range, plot=False):
    '''
    Find the multiplier between short and long exposure times

    Inputs:
        short_exptime: 2D array of short exposure time image
        long_exptime: 2D array of long exposure time image
        fit_range: tuple of (min, max) values for considered MSB
    Returns:
        float: the multiplier between short and long exposure times
    '''
    # replace unselected tiles marked as nan to zeros
    long_exptime = np.nan_to_num(long_exptime, posinf=np.max(long_exptime[np.isfinite(long_exptime)]))
    short_exptime = np.nan_to_num(short_exptime, posinf=np.max(short_exptime[np.isfinite(short_exptime)]))
    
    div_long_exptime = copy.deepcopy(long_exptime)
    div_long_exptime[long_exptime == 0] = np.nan
    diff = short_exptime / div_long_exptime
    diff[long_exptime > fit_range[1]] = np.nan
    diff[long_exptime < fit_range[0]] = np.nan
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(diff, cmap='viridis', origin='lower')
        plt.colorbar(label='Short Exptime / Long Exptime')
        plt.annotate(f'Mean Intensity Ratio: {np.nanmean(diff):.2f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, color='black', ha='left', va='top')
        plt.annotate(f'No. of pixels considered: {np.count_nonzero(~np.isnan(diff))}', xy=(0.05, 0.90), xycoords='axes fraction', fontsize=12, color='black', ha='left', va='top')
        plt.show()
    return 1/np.nanmean(diff)


def combine(long_exptime, short_exptime, threshold = 3500, global_gain_hg=1, short_exptime_mult=1806):
    # replace unselected tiles marked as nan to zeros
    long_exptime = np.nan_to_num(long_exptime, posinf=np.max(long_exptime[np.isfinite(long_exptime)]))
    short_exptime = np.nan_to_num(short_exptime, posinf=np.max(short_exptime[np.isfinite(short_exptime)]))

    # if LG > x, then HG, else LG
    lg_pixels           = (long_exptime > threshold) | (long_exptime == 0)
    lg_mask             = lg_pixels * 1.0
    # plt.imshow(lg_mask, cmap='gray', origin='lower')
    # using a more smooth transition between LG and HG pixels
    lg_mask             = cv2.GaussianBlur(lg_mask,(1,1), 1)
    hg_mask             = 1 - lg_mask


    combined            = long_exptime * global_gain_hg * hg_mask
    combined           += short_exptime * global_gain_hg * short_exptime_mult * lg_mask


    # replaced zeros with nan
    # combined[combined == 0] = np.nan

    return combined

def merge_exposure(imarray, option='simple'):
    if option == 'simple':
        im_merge_new = np.where(np.isfinite(imarray[2]), imarray[2], imarray[1])
        im_merge_new = np.where(np.isfinite(im_merge_new), im_merge_new, imarray[0])
    elif option == 'blend':
        # find maximum finite values of imarray[0], imarray[1], and imarray[2], there are np.inf values in those array
        max_0 = np.max(imarray[0][np.isfinite(imarray[0])])
        max_1 = np.max(imarray[1][np.isfinite(imarray[1])])
        max_2 = np.max(imarray[2][np.isfinite(imarray[2])])
        # define fit range as 80 - 99 % of the maximum value of the long exposure image
        fit_range_1s_10s = (max_2 * 0.8, max_2 * 0.99)
        fit_range_01s_1s = (max_1 * 0.8, max_1 * 0.99)

        # define threshold as 90% of maximum value of the long exposure image
        threshold_1s_10s = 0.99*max_2
        threshold_01s_1s = 0.99*max_1

        mult_1s_10s = find_multiplier(imarray[1], imarray[2], fit_range=fit_range_1s_10s, plot=True)
        # print(mult_1s_10s)
        mult_01s_1s = find_multiplier(imarray[0], imarray[1], fit_range=fit_range_01s_1s, plot=True)
        # print(mult_01s_1s)
        im_merge_1s_10s = combine(imarray[2], imarray[1], threshold=threshold_1s_10s, short_exptime_mult=mult_1s_10s)
        im_merge_new = combine(im_merge_1s_10s, imarray[0], threshold=threshold_01s_1s, short_exptime_mult=mult_01s_1s)

    return im_merge_new

    
