import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import glob
import os
import urllib.request
from pathlib import Path

'''
Might try to migrate this into sunpy mainpackage instead???
'''
def generate_colormap():
    mycwd = os.getcwd()
    os.chdir(Path(__file__).resolve().parent)
    os.makedirs('./cmap_file', exist_ok=True)
    cmap_filelist_num = len(glob.glob('./cmap_file/*.txt'))
    if cmap_filelist_num < 5:
        cmap_namelist = ['wb_colormap.txt', 'fe_colormap.txt', 'he_colormap.txt', 'p_colormap.txt', 'ne_colormap.txt']
        print('Downloading Colourmap files from SIDC....')
        for i in range(len(cmap_namelist)):
            urllib.request.urlretrieve('https://www.sidc.be/sites/default/files/2025-07/'+cmap_namelist[i], './cmap_file/'+cmap_namelist[i])
    
    wb_reffile = np.loadtxt('./cmap_file/wb_colormap.txt')
    wb_cmap = LinearSegmentedColormap.from_list('ASPIICS Wideband', wb_reffile)
    wb_cmap.set_bad(color='tab:grey')
    mpl.colormaps.register(cmap=wb_cmap)

    fe_reffile = np.loadtxt('./cmap_file/fe_colormap.txt')
    fe_cmap = LinearSegmentedColormap.from_list('ASPIICS Fe XIV', fe_reffile)
    fe_cmap.set_bad(color='tab:grey')
    mpl.colormaps.register(cmap=fe_cmap)

    he_reffile = np.loadtxt('./cmap_file/he_colormap.txt')
    he_cmap = LinearSegmentedColormap.from_list('ASPIICS He I', he_reffile)
    he_cmap.set_bad(color='tab:grey')
    mpl.colormaps.register(cmap=he_cmap)

    pb_reffile = np.loadtxt('./cmap_file/p_colormap.txt')
    pb_cmap = LinearSegmentedColormap.from_list('ASPIICS pB', pb_reffile)
    pb_cmap.set_bad(color='tab:grey')
    mpl.colormaps.register(cmap=pb_cmap)

    ne_reffile = np.loadtxt('./cmap_file/ne_colormap.txt')
    ne_cmap = LinearSegmentedColormap.from_list('ASPIICS ne', ne_reffile)
    ne_cmap.set_bad(color='tab:grey')
    mpl.colormaps.register(cmap=ne_cmap)

    os.chdir(mycwd) 

if __name__ == "__main__":
    generate_colormap()
    print('Generating and Registering Colourmap for PROBA3_ASPIICS')