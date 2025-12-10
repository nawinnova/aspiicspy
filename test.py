# write test for aspiicspy package using pytest

import aspiicspy
import pytest

def test_import_aspiicspy():
    assert aspiicspy is not None
    assert hasattr(aspiicspy, 'read_L2')
    assert hasattr(aspiicspy, 'generate_colormap')

def test_generate_colormap():
    colormaps = aspiicspy.generate_colormap.generate_colormap()
    assert isinstance(colormaps, dict)
    assert 'ASPIICS pB' in colormaps
    assert 'ASPIICS Wideband' in colormaps

def test_read_L2():
    # Use a sample fits file path for testing
    sample_fits_path = '/path/to/sample_l2_file.fits'  # Replace with an actual test file path
    try:
        aspiics_map = aspiicspy.read_L2.read_aspiics_sunpy(sample_fits_path)
        assert aspiics_map is not None
        assert hasattr(aspiics_map, 'meta')
        assert 'filter' in aspiics_map.meta
    except FileNotFoundError:
        pytest.skip("Sample L2 FITS file not found, skipping read_L2 test.")

if __name__ == "__main__":
    pytest.main([__file__])

