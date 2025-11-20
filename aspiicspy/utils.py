import numpy as np
import sunpy.map


def merge_3exposure_maps(map1, map2, map3):
    """
    Merge three SunPy map objects by averaging their pixel values.

    Parameters
    ----------
    map1, map2, map3 : sunpy.map.Map
        The three map objects to merge. 
        Should have the same dimensions
        as well as the overexposed pixels marked as NaNs.

    Returns
    -------
    sunpy.map.Map
        A new map object containing the merged data.
    """
    # Ensure all maps have the same shape
    if map1.data.shape != map2.data.shape or map1.data.shape != map3.data.shape:
        raise ValueError("All maps must have the same dimensions to merge.")
    # create list of map and sorted by exposure time (ascending)
    map_list = [map1, map2, map3]
    map_list.sort(key=lambda x: x.meta['exptime'])
    # Merge the data by replacing NaNs of map with shorter exposure time
    # note that there will be discontinuity at the boundary where pixel values are taken from different exposure times
    merged_exposure1 = np.where(np.isfinite(map_list[2].data), map_list[2].data, map_list[1].data)
    merged_exposure = np.where(np.isfinite(merged_exposure1), merged_exposure1, map_list[0].data)

    # Create a new map object with the merged data
    merged_map = sunpy.map.Map(merged_exposure, map_list[1].meta)

    return merged_map