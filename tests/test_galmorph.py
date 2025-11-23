import morphology

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
    try:
        test_inc = morphology.FindInc2(test_eta, 0.0, test_B, test_C, test_D)
    except ZeroDivisionError:
        pass
    else:
        assert False, "Expected ZeroDivisionError for A=0"
    #test B = 0 raises ZeroDivisionError
    try:
        test_inc = morphology.FindInc2(test_eta, test_A, 0.0, test_C, test_D)
    except ZeroDivisionError:
        pass
    else:
        assert False, "Expected ZeroDivisionError for B=0"
    #testing non-finite inputs raises ValueError
    test_vals = [np.nan, np.inf, -np.inf]
    for val in test_vals:
        try:
            test_inc = morphology.FindInc2(val, test_A, test_B, test_C, test_D)
        except ValueError:
            pass
        else:
            assert False, "Expected ValueError for non-finite eta"
        try:
            test_inc = morphology.FindInc2(test_eta, val, test_B, test_C, test_D)
        except ValueError:
            pass
        else:
            assert False, "Expected ValueError for non-finite A"
        try:
            test_inc = morphology.FindInc2(test_eta, test_A, val, test_C, test_D)
        except ValueError:
            pass
        else:
            assert False, "Expected ValueError for non-finite B"
        try:
            test_inc = morphology.FindInc2(test_eta, test_A, test_B, val, test_D)
        except ValueError:
            pass
        else:
            assert False, "Expected ValueError for non-finite C"
        try:
            test_inc = morphology.FindInc2(test_eta, test_A, test_B, test_C, val)
        except ValueError:
            pass
        else:
            assert False, "Expected ValueError for non-finite D"
        
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
    try:
        morphology.determine_background_radius(R, I, noisefloor, window_size=0)
    except AssertionError:
        pass
    else:
        assert False, "Expected AssertionError for zero window_size" 
    try:
        morphology.determine_background_radius(R, I, noisefloor, window_size=100.0001)
    except AssertionError:
        pass
    else:
        assert False, "Expected AssertionError for non-integer window_size"
    try:
        morphology.determine_background_radius(R, I, noisefloor, window_size=1001)
    except AssertionError:
        pass
    else:
        assert False, "Expected AssertionError for window_size larger than R, I length"
    #testing edge case where no values fall below noisefloor
    I_ = 0*I
    maxrad = morphology.determine_background_radius(R, I_, noisefloor, window_size=window_size)
    assert maxrad == np.nanmax(R), "Expected maxrad to be np.nanmax(R) when no values fall below noisefloor"
    I_ = I.copy()
    I_[0] = -10  # make first value below noisefloor be at R = 0
    maxrad = morphology.determine_background_radius(R, I_, noisefloor)
    assert maxrad != 0.0, "Maxrad should not be 0.0 even if first value is below noisefloor, default to second value or nanmax(R)"
    #testing cases where R, I are not 1D arrays or are different sizes
    try:
        morphology.determine_background_radius(R.reshape(1000,1), I, noisefloor, window_size=window_size)
    except AssertionError:
        pass
    else:
        assert False, "Expected AssertionError for non-1D R"

    try:
        morphology.determine_background_radius(R, I.reshape(1000,1), noisefloor)
    except AssertionError:
        pass
    else:
        assert False, "Expected AssertionError for non-1D I"

    try:
        morphology.determine_background_radius(R[:-1], I, noisefloor)
    except AssertionError:
        pass
    else:
        assert False, "Expected AssertionError for mismatched lengths of R and I"

def test_galaxymorphology():
    '''test galaxymorphology function'''
    #test with a sample FITS file and mock data

#need to write test for galaxymorphology function with a sample FITS file and mock data...


def test_galmorph_version():
    """Test that the version string is correctly set."""
    import galaxyinclinations
    assert isinstance(galaxyinclinations.__version__, str)


