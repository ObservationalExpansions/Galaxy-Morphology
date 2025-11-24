from flex import FLEX

from scipy.stats import linregress
from astropy.io import fits
from astropy.wcs import WCS

import numpy as np

def FindInc2(eta, A, B, C, D):
    """
    Return the inclination angle (in degrees) that corresponds to an observed value eta
    by inverting a cosine model.
    This function inverts the model
        eta = A * cos(B * i_rad + C) + D
    for the inclination i (returned in degrees, constrained to [0, 90]). The routine
    handles numerical robustness and the cosine symmetry/periodicity by:
      - treating a special-case saturation (if eta > 0.5 the function returns 90.0),
      - computing a safe inverse argument x = (eta - D) / A and clipping it to [-1, 1],
      - forming the two symmetric arccos solutions and converting them from radians to
        degrees,
      - reducing candidate angles by the cosine period and folding into [0, 90],
      - evaluating the forward model for both candidates and returning the one with
        the smallest absolute residual to the input eta.
    Parameters
    ----------
    eta : float
        Observed quantity to invert (scalar). Expected to be finite.
    A : float
        Amplitude of the cosine term. Must be non-zero (division by zero otherwise).
    B : float
        Angular scaling factor used in the cosine argument. Must be non-zero.
        B and C are used such that B * theta + C is the argument to cos, where theta
        is in radians.
    C : float
        Phase offset (radians) passed directly into the cosine argument.
    D : float
        Vertical offset of the cosine model.
    Returns
    -------
    float
        Inclination angle in degrees, constrained to the interval [0.0, 90.0].
    Raises
    ------
    ZeroDivisionError
        If A == 0 or B == 0 (these produce divisions by zero in the inversion).
    ValueError
        If inputs are not finite scalars.
    Notes
    -----
    - All trigonometric operations inside the function use radians. Provide C in
      radians and interpret B consistently (so that B * i_rad + C is dimensionally correct).
    - The implementation clips the inverse cosine argument to [-1, 1] to avoid NaNs due
      to floating-point rounding and then selects the candidate angle that best
      reproduces the observed eta.
    - The function is intended for scalar inputs (not vectorized across arrays).
    """
    # Validate inputs explicitly (use exceptions rather than `assert` so checks
    # remain active even if Python is run with optimizations (-O)).
    finite_checks = [np.isfinite(eta), np.isfinite(A), np.isfinite(B), np.isfinite(C), np.isfinite(D)]
    if not all(finite_checks):
        raise ValueError("All inputs must be finite scalars")
    scalar_checks = [np.isscalar(eta), np.isscalar(A), np.isscalar(B), np.isscalar(C), np.isscalar(D)]
    if not all(scalar_checks):
        raise ValueError("All inputs must be scalar values")
    if eta>0.5:
        inc=90
    else:
        # 1) Safe inverse argument (A can be negative)
        x = np.clip((eta - D) / A, -1.0, 1.0)

        # 2) Two branches from cos symmetry, all in radians
        a = np.arccos(x)                # in [0, pi]
        i1 = (a - C) / B
        i2 = (-a - C) / B

        # 3) Convert candidates to degrees
        cand = np.rad2deg(np.array([i1, i2]))

        # 4) Reduce by the period in degrees: P = 2π/|B| (then to degrees)
        P = np.rad2deg(2*np.pi / abs(B))
        cand = cand % P                 # map into one period

        # 5) Fold into [0, 180], then reflect >90 across 90 to get [0, 90]
        cand = np.where(cand > 180.0, cand - 180.0, cand)
        cand = np.where(cand > 90.0, 180.0 - cand, cand)

        # 6) Pick the candidate that best reproduces eta (keeps you on the line)
        def model(i_deg):
            return A*np.cos(B*np.deg2rad(i_deg) + C) + D

        errs = np.abs(model(cand) - eta)
        inc=float(cand[np.argmin(errs)])
    return inc


def determine_background_radius(R, I, noisefloor, window_size=100):
    """
    Determine the radius at which the azimuthally-averaged intensity profile falls to a specified noise floor.
    Comment: Searches the smoothed (moving-average) radial intensity profile for the first radius where log(intensity) < noisefloor.
    Parameters
    ----------
    R : array-like, shape (N,)
        Radial coordinates for each pixel/sample. Will be sorted internally; values may be any real numbers.
    I : array-like, shape (N,)
        Intensity values corresponding to R. Values should be positive for the logarithm; zeros or negatives replaced with NaN to avoid -inf or runtime warnings.
    noisefloor : float
        Threshold applied to np.log(smoothed_intensity). The function returns the radius corresponding to the first smoothed bin whose log(intensity) is below this threshold.
    window_size : int, optional
        Size of the moving-average window used for smoothing the intensity profile. Default is 100 samples
    Returns
    -------
    float
        maxrad: The radius at which the smoothed log-intensity first drops below `noisefloor`. If no such radius is found or an error occurs (e.g., due to invalid values), the function falls back to the maximum of R (np.nanmax(R)).
    Behavior and notes
    ------------------
    - The inputs R and I are sorted by R internally before computing the moving average.
    - A fixed moving-average window size of 100 samples is used (hyperparameter). The convolution uses mode='valid', so the smoothed arrays are shorter than the original by window_size-1.
    - The returned radius is taken from the smoothed radius array corresponding to the first bin where np.log(smoothed_intensity) < noisefloor.
    - If smoothed intensities contain non-positive values, np.log will yield -inf or NaN; such cases may trigger the fallback behavior.
    - The function catches exceptions arising during the threshold search and returns np.nanmax(R) as a conservative fallback.
    Example
    -------
    >>> import numpy as np
    >>> R = np.linspace(0, 100, 1000)
    >>> I = np.exp(-R/20) + 0.001 * np.random.randn(1000)
    >>> determine_background_radius(R, I, noisefloor=-7.0)
    """
    assert isinstance(R, np.ndarray), "R must be a numpy array"
    assert isinstance(I, np.ndarray), "I must be a numpy array"
    if R.ndim != 1 or I.ndim != 1:
        raise ValueError("R and I must be 1D arrays")
    if len(R) != len(I):
        raise ValueError("R and I must have the same length")
    if np.nanmin(R) < 0:
        raise ValueError("R must be exclusively non-negative values")
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if window_size > len(R):
        raise ValueError("window_size must be less than or equal to the length of R and I")

    # sort by radius for fit
    rindx = R.argsort()
    R_sorted = R[rindx]
    I_sorted = I[rindx]

    # generate a moving average
    means_R = np.convolve(R_sorted, np.ones(window_size)/window_size, mode='valid')
    means_I = np.convolve(I_sorted, np.ones(window_size)/window_size, mode='valid')
    means_I = np.where(means_I <= 0, np.nan, means_I)  # avoid log of non-positive numbers
    # set the floor for noise
    # if this fails, fall back to R.max()
    try:
        maxrad = means_R[np.where(np.log(means_I) < noisefloor)[0][0]]
        # handle edge case where first value is zero, try the next bigger radius
        if maxrad == 0.0:
            try:
                maxrad = means_R[np.where(np.log(means_I) < noisefloor)[0][1]]
            except:
                maxrad = np.nanmax(R)
    except:
        maxrad = np.nanmax(R)

    return maxrad


def galaxymorphology(file,galaxy=None,data=None,noisefloor=-5.):

    if data is not None and galaxy is None:
        raise ValueError("If 'data' is provided, 'galaxy' must also be defined.")
    
    
    with fits.open(file) as hdulist:
        image_data = hdulist[1].data

        if data is not None:
            ra = data['RA_LEDA'][data['GALAXY']==galaxy]
            dec = data['DEC_LEDA'][data['GALAXY']==galaxy]

            wcs = WCS(hdulist[1].header)
            pixel_coords = wcs.world_to_pixel_values(ra, dec)
        
    galaxy_name = file.split("-")[0]
    h, w = image_data.shape

    # set default
    if data is None:
        cx, cy = w//2, h//2
    else:   
        cx, cy = pixel_coords[0], pixel_coords[1]

    # this choice of radius is a hyperparameter -- we may want to tune it
    radius=h//4

    y, x = np.indices((h,w))

    # compute the cartesian pixel coordinates relative to center
    X2,Y2 = (x-cx), (y-cy)

    # created 1d radial profile
    R = np.sqrt(X2**2 + Y2**2).ravel()
    I = image_data.ravel()

    valid_ = (I > 0) & (R <= radius)
    R_valid_ = R[valid_]
    I_valid_ = I[valid_]

    maxrad = determine_background_radius(R_valid_, I_valid_, noisefloor)
    
    valid = (I > 0) & (R <= maxrad)
    R_valid = R[valid]
    logI_valid = np.log(I[valid])
   
    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(R_valid, logI_valid)
    assert slope != 0, "Slope of linear regression is zero, cannot compute scale length."
    scale_length = -1 / slope
    assert scale_length > 0, "Computed scale length is not positive."

    # we might want to check for bogus values here: but we can catch these on return as well
    
    # compute the cylindrical coordinates
    R,P = np.sqrt(X2**2 + Y2**2).flatten(), np.arctan2(Y2, X2).flatten()
    I = image_data.flatten()
    galaxy_pixels = (R <= maxrad)

    mmax, nmax = 2, 10

    afacs = np.array([0.75,1.0, 1.5, 2.0, 2.5, 3.0])
    etalist = np.zeros([3,len(afacs)])
    palist = np.zeros([2,len(afacs)])
    for i, afac in enumerate(afacs):
        L = FLEX(scale_length*afac,mmax,nmax, R[galaxy_pixels], P[galaxy_pixels], mass=I[galaxy_pixels])

        A2 = np.linalg.norm(np.sqrt(L.coscoefs[2]**2 + L.sincoefs[2]**2))
        A1 = np.linalg.norm(np.sqrt(L.coscoefs[1]**2 + L.sincoefs[1]**2))
        A0 = np.linalg.norm(L.coscoefs[0])

        etalist[0,i]     = A0
        etalist[1,i]     = A1
        etalist[2,i]     = A2
        
        # if you take the average, you will smear out some signal
        palist[0,i]      = np.nansum(np.sqrt(L.coscoefs[2]**2 + L.sincoefs[2]**2)*np.arctan2(L.sincoefs[2], L.coscoefs[2]))/np.nansum(np.sqrt(L.coscoefs[2]**2 + L.sincoefs[2]**2))  # in radians

        # try PA that is just n=0?
        palist[1,i] = np.arctan2(L.sincoefs[2][0], L.coscoefs[2][0])  # in radians


    return etalist, palist, scale_length, galaxy_name, radius, maxrad
    
