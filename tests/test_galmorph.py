from flex import FLEX
import pytest
from scipy.stats import linregress
from astropy.io import fits
from astropy.wcs import WCS

import numpy as np
from galaxyinclinations import morphology

def test_FindInc2():
    '''test finding the inclination'''
    #test eta > 0.5 returns 90
    test_A= 0.3
    test_B= 1.0
    test_C= 0.0
    test_D= 0.2
    test_eta = 0.3
    test_inc = morphology.FindInc2(0.5000000001, test_A, test_B, test_C, test_D)
    assert test_inc == 90.0
    #test A = 0 raises ZeroDivisionError
    with pytest.raises(ZeroDivisionError):
        test_inc = morphology.FindInc2(test_eta, 0.0, test_B, test_C, test_D)
    #test B = 0 raises ZeroDivisionError
    with pytest.raises(ZeroDivisionError):
        test_inc = morphology.FindInc2(test_eta, test_A, 0.0, test_C, test_D)
    #testing non-finite inputs raises ValueError
    with pytest.raises(ValueError):
        test_inc = morphology.FindInc2(np.inf, test_A, test_B, test_C, test_D)
        test_inc = morphology.FindInc2(test_eta, np.inf, test_B, test_C, test_D)
        test_inc = morphology.FindInc2(test_eta, test_A, np.inf, test_C, test_D)
        test_inc = morphology.FindInc2(test_eta, test_A, test_B, np.inf, test_D)
        test_inc = morphology.FindInc2(test_eta, test_A, test_B, test_C, np.inf)
        #now nans 
        test_inc = morphology.FindInc2(np.nan, test_A, test_B, test_C, test_D)
        test_inc = morphology.FindInc2(test_eta, np.nan, test_B, test_C, test_D)
        test_inc = morphology.FindInc2(test_eta, test_A, np.nan, test_C, test_D)
        test_inc = morphology.FindInc2(test_eta, test_A, test_B, np.nan, test_D)
        test_inc = morphology.FindInc2(test_eta, test_A, test_B, test_C, np.nan)
        
    #test some normal case
    test_inc = morphology.FindInc2(test_eta, test_A, 2, .9, test_D)
    assert isinstance(test_inc,float), "Expected float output for valid inputs"


def test_determine_background_radius():
    '''test determining background radius'''
    R = np.linspace(0, 100, 1000)
    I = np.exp(-R/20) + 0.001 * np.random.randn(1000)
    noisefloor = -7.0
    window_size = 100
    #testing some normal values
    maxrad = morphology.determine_background_radius(R, I, noisefloor, window_size=window_size)
    assert isinstance(maxrad, float), "Expected float output"
    assert 0 <= maxrad <= R.max(), "Expected maxrad to be within the range of R"
    #testing invalid window_size values
    with pytest.raises(ValueError):
        morphology.determine_background_radius(R, I, noisefloor, window_size=0)
        morphology.determine_background_radius(R, I, noisefloor, window_size=100.01)
        morphology.determine_background_radius(R, I, noisefloor, window_size=1001)
    #testing edge case where no values fall below noisefloor
    I_ = np.ones_like(I)
    maxrad = morphology.determine_background_radius(R, I_, noisefloor, window_size=window_size)
    assert maxrad == np.nanmax(R), "Expected maxrad to be np.nanmax(R) when no values fall below noisefloor"
    I_ = I.copy()
    I_[0] = -10  # make first value below noisefloor be at R = 0
    maxrad = morphology.determine_background_radius(R, I_, noisefloor)
    assert maxrad != 0.0, "Maxrad should not be 0.0 even if first value is below noisefloor, default to second value or nanmax(R)"
    #testing cases where R, I are not 1D arrays or are different sizes
    with pytest.raises(ValueError):
        morphology.determine_background_radius(R.reshape(1000,1), I, noisefloor, window_size=window_size)
        morphology.determine_background_radius(R, I.reshape(1000,1), noisefloor)
        morphology.determine_background_radius(R[:-1], I, noisefloor)

#need to write test for galaxymorphology function with a sample FITS file and mock data...


def test_galmorph_version():
    """Test that the version string is correctly set."""
    import galaxyinclinations
    assert isinstance(galaxyinclinations.__version__, str)


