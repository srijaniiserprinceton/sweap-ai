# This code is a reproduction of an IDL code written by Dr. Mike Stevens.
#  
# --- SPC Effective Area Modeling ---
#
# Usage: 
# areas = psp_swp_spc_coldspot(phi, theta)
#
# Inputs:
#
# phi   = [scalar or 1 x N array] of horizontal incidence angles [DEGREES]
# theta = [scalar or 1 x N array] of vertical incidence angles [DEGREES]
#
# Cup geometry is set to be a global dictionary. This can be overwritten 
# if needed. 
#
# Return:
#
# 4 x N element array of effective collecting areas A [mm**2]. 
# The measured charge flux upon the respective sensor quadrants is I ~ nqvA

# Load in packages
import numpy as np
import math
import sys
import time

# Note: in Python floating point values are defined to be double precision.
glbl_param_dict = {
                    're': float(39.8),
                    'rl': float(10.86),
                    'rc': float(23.94),
                    'ze': float(70.5),
                    'zl': float(16.9),
                    'epsilon': float(0.36),
                    'dz_coll': float(1.981)
                  }

# Pre-calculate defined arctan values
# arctan2 is used to give a value in range -pi to pi
aL = np.degrees(np.arctan2((glbl_param_dict['re'] - glbl_param_dict['rl']), 
                           (glbl_param_dict['ze'] - glbl_param_dict['zl']))) 

aC = np.degrees(np.arctan2((glbl_param_dict['rc'] - glbl_param_dict['rl']),
                            glbl_param_dict['zl']))

amax = np.degrees(np.arctan2((glbl_param_dict['rc'] + glbl_param_dict['re']),
                              glbl_param_dict['ze']))

glbl_param_dict['aL'] = aL
glbl_param_dict['aC'] = aC
glbl_param_dict['amax'] = amax


# Integrate the area under a semicircle in the (x,y) plance with respect to x
# over a finite domain [x0, x1]


def int_uppersemicircle_dx(r=1, xc=0, yc=0, x0=None, x1=None):
    #_______________________________________________________________________
    # This function sets the initial parameters if no other values are given
    # 
    # r  - is defaulted to a unit circle
    # xc - defines the x offeset to be 0
    # yc - defines the y offeset to be 0
    # x0 - set to be None
    # x1 - set to be None
    #
    #_______________________________________________________________________

    if x0 is None: x0 = xc - np.abs(r)
    if x1 is None: x1 = xc + np.abs(r)

    if x1 == x0: return 0

    # Swap integration bounds if needed 
    # This can be set to have only one if statement...
    if x1 < x0: 
        lower = x1
        upper = x0
    else:
        lower = x0
        upper = x1

    # Check if range is entirely outside or the circle
    if (lower >= (xc + np.abs(r))) or (upper <= xc - np.abs(r)): return 0
    
    # if bound is beyond circle domain
    if (lower < (xc - np.abs(r))): lower = xc - np.abs(r)
    if (upper > (xc + np.abs(r))): upper = xc + np.abs(r)

    num_upper = upper - xc
    num_lower = lower - xc

    # This is a segment of code taken directly from Mike's code
    # Check that the agruments are not negative or very small numbers
    # Therefore we use an ad hoc tolerance of -1e-9
    denom_uparg = -xc**2 + r**2 + 2.*xc*upper - upper**2
    denom_loarg = -xc**2 + r**2 + 2.*xc*lower - lower**2

    # Assuming this section of code wants the logical integer value not True or False
    if denom_uparg > (-1e-9): denom_uparg = np.maximum(denom_uparg, 0)
    if denom_loarg > (-1e-9): denom_loarg = np.maximum(denom_loarg, 0)

    if (r**2 + 2*xc*upper) == (upper**2 + xc**2): denom_upper = 0
    else: denom_upper = np.sqrt(denom_uparg)

    if (r**2 + 2*xc*lower) == (lower**2 + xc**2): denom_lower = 0
    else: denom_lower = np.sqrt(denom_loarg)

    arg_upper = yc*upper + 0.5*num_upper*denom_upper + 0.5*(r**2)*np.arctan2(num_upper, denom_upper)
    arg_lower = yc*lower + 0.5*num_lower*denom_lower + 0.5*(r**2)*np.arctan2(num_lower, denom_lower)

    if (np.any(np.isnan(arg_upper - arg_lower)) is True) or (arg_upper == arg_lower): return 0

    return arg_upper - arg_lower    

def int_lowersemicircle_dx(r=1, xc=0, yc=0, x0=None, x1=None):
    # This funciton will likely be obsolete as we can pass the keyword lower/upper
    #_______________________________________________________________________
    # This function sets the initial parameters if no other values are given
    # 
    # r  - is defaulted to a unit circle
    # xc - defines the x offeset to be 0
    # yc - defines the y offeset to be 0
    # x0 - set to be None
    # x1 - set to be None
    #
    #_______________________________________________________________________

    if x0 is None: x0 = xc - np.abs(r)
    if x1 is None: x1 = xc + np.abs(r)

    if x1 == x0: return 0

    # Swap integration bounds if needed 
    # This can be set to have only one if statement...
    if x1 < x0: 
        lower = x1
        upper = x0
    else:
        lower = x0
        upper = x1

    if (lower >= (xc + np.abs(r))) or (upper <= xc - np.abs(r)): return 0
    
    if (lower < (xc - np.abs(r))): lower = xc - np.abs(r)
    if (upper > (xc + np.abs(r))): upper = xc + np.abs(r)

    num_upper = xc - upper
    num_lower = xc - lower

    # This is a segment of code taken directly from Mike's code
    # Check that the agruments are not negative or very small numbers
    # Therefore we use an ad hoc tolerance of -1e-9
    denom_uparg = -xc**2 + r**2 + 2.*xc*upper - upper**2
    denom_loarg = -xc**2 + r**2 + 2.*xc*lower - lower**2

    # Assuming this section of code wants the logical integer value not True or False
    if denom_uparg > (-1e-9): denom_uparg = np.maximum(denom_uparg, 0)
    if denom_loarg > (-1e-9): denom_loarg = np.maximum(denom_loarg, 0)

    if (r**2 + 2*xc*upper) == (upper**2 + xc**2): denom_upper = 0
    else: denom_upper = np.sqrt(denom_uparg)

    if (r**2 + 2*xc*lower) == (lower**2 + xc**2): denom_lower = 0
    else: denom_lower = np.sqrt(denom_loarg)

    arg_upper = yc*upper + 0.5*num_upper*denom_upper + 0.5*(r**2)*np.arctan2(num_upper, denom_upper)
    arg_lower = yc*lower + 0.5*num_lower*denom_lower + 0.5*(r**2)*np.arctan2(num_lower, denom_lower)

    if (np.any(np.isnan(arg_upper - arg_lower)) is True) or (arg_upper == arg_lower): return 0

    return arg_upper - arg_lower

# NOTE will replace the above code with the following in future revision
def int_semicircle_dx(r=1, xc=0, yc=0, x0=None, x1=None, DOMAIN='upper'):
    #_______________________________________________________________________
    # This function sets the initial parameters if no other values are given
    # 
    # r  - is defaulted to a unit circle
    # xc - defines the x offeset to be 0
    # yc - defines the y offeset to be 0
    # x0 - set to be None
    # x1 - set to be None
    #
    #_______________________________________________________________________

    if x0 is None: x0 = xc - np.abs(r)
    if x1 is None: x1 = xc + np.abs(r)

    if x1 == x0: return 0

    # Swap integration bounds if needed 
    # This can be set to have only one if statement...
    if x1 < x0: 
        lower = x1
        upper = x0
    else:
        lower = x0
        upper = x1

    if (lower >= (xc + np.abs(r))) or (upper <= xc - np.abs(r)): return 0
    
    if (lower < (xc - np.abs(r))): lower = xc - np.abs(r)
    if (upper > (xc + np.abs(r))): upper = xc - np.abs(r)

    if DOMAIN == "upper":
        num_upper = upper - xc
        num_lower = lower - xc
    if DOMAIN == "lower":
        num_upper = xc - upper
        num_lower = xc - lower

    # This is a segment of code taken directly from Mike's code
    # Check that the agruments are not negative or very small numbers
    # Therefore we use an ad hoc tolerance of -1e-9
    denom_uparg = -xc**2 + r**2 + 2.*xc*upper - upper**2
    denom_loarg = -xc**2 + r**2 + 2.*xc*lower - lower**2

    # Assuming this section of code wants the logical integer value not True or False
    if denom_uparg > (-1e-9): denom_uparg = np.minimum(denom_uparg, 0)
    if denom_loarg > (-1e-9): denom_loarg = np.minimum(denom_loarg, 0)

    if (r**2 + 2*xc*upper) == (upper**2 + xc**2): denom_upper = 0
    else: denom_upper = np.sqrt(denom_uparg)

    if (r**2 + 2*xc*lower) == (lower**2 + xc**2): denom_lower = 0
    else: denom_lower = np.sqrt(denom_loarg)

    arg_upper = yc*upper + 0.5*num_upper*denom_upper + 0.5*(r**2)*np.arctan2(num_upper, denom_upper)
    arg_lower = yc*lower + 0.5*num_lower*denom_lower + 0.5*(r**2)*np.arctan2(num_lower, denom_lower)

    if (np.all(np.isfinite(arg_upper - arg_lower)) is False) or (arg_upper == arg_lower): return 0

    return arg_upper - arg_lower    

def intersection_of_circles(x1, y1, r1, x2, y2, r2):

    # distance between circle centers
    d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # if they are the same circle return extreme values
    if (d == 0) and (np.abs(r1) == np.abs(r2)): return [x1 - r1, x1 + r1]

    # if the circles are too far apart, there is no intersection
    if (d > (np.abs(r1) + np.abs(r2))): return [np.nan, np.nan]

    # if one circle is entirely within the other, there are no intersections
    if (d < np.abs(np.abs(r2) - np.abs(r1))): return [np.nan, np.nan]

    # distance of one circle center to midpoint between the two circles
    a = (r1**2 - r2**2 + d**2)/(2.*d)

    # distance from intersection to midpoint of the circles
    h = np.sqrt(r1**2 - a**2)

    # Location of the midpoint between the two circles
    x3 = x1 + a*(x2-x1)/d
    y3 = y1 + a*(y2-y1)/d

    # x at the two intersections
    x4 = x3 + h*(y2-y1)/d
    x5 = x3 - h*(y2-y1)/d

    return [np.minimum(x4, x5), np.maximum(x4, x5)]

def area_of_intersection(x1, x2, y1, y2, r1, r2):
    # THIS CODE IS NOT USED! COPIED FOR COMPLETNESS
    intercepts = intersection_of_cicles(x1=x1, x2=x2, y1=y1, y2=y2, r1=r1, r2=r2)
    
    # This is just giving values between 0 and 1 for xmax and xmin?
    xmax = np.minimum((x1 + np.abs(r1)), (x2 + np.abs(r2)))
    xmin = np.maximum((x1 - np.abs(r1)), (x2 - abs(r2)))

    # we need to make sure that the intercepts values are added to the array
    # and not the array itself. The size of xpoints should be 4!
    xpoints = [xmin, *intercepts, xmax]

    # This reduces the number of points to be only finite valued
    xi = xpoints[np.isfinite(xpoints)]

    area = 0

    # notice the idl code originally has a n_elements - 2 however,
    # we need to modify this section so that we go to the second to last element
    # in the array.
    for i in range(0, len(xi)-1):
        low1  = int_lowersemicircle_dx(r=r1, xc = x1, yc = y1, x0 = xi[i], x1 = xi[i+1])
        low2  = int_lowersemicircle_dx(r=r2, xc = x2, yc = y2, x0 = xi[i], x1 = xi[i+1])
        high1 = int_uppersemicircle_dx(r=r1, xc = x1, yc = y1, x0 = xi[i], x1 = xi[i+1])
        high2 = int_uppersemicircle_dx(r=r1, xc = x2, yc = y2, x0 = xi[i], x1 = xi[i+1])

        thisarea = np.minimum(high1, high2) - np.maximum(low1, low2)
        area = area + np.maximum(thisarea, 0)

    return area

def circle_x_at_yeq(y, xc, yc, r):
    arg = r**2 - (y - yc)**2
    
    if arg < 0: return [np.nan, np.nan]

    return [xc - np.sqrt(r**2 - (y - yc))**2, xc + np.sqrt(r**2 - (y - yc)**2)]

def circle_xintercepts(xc, yc, r):
    arc = r**2 - yc**2

    if arg < 0: return [np.nan, np.nan]

    return [xc - np.sqrt(arg), xc + np.sqrt(arg)]

def quadrant_area_of_intersection(x1, x2, y1, y2, r1, r2, x3=None, y3=None, r3=None, epsilon=None):
    
    # Replace the case tabel in the IDL code. 
    # If no values given set the epsilon values to 0.
    if epsilon is None:
        eps_x = 0
        epx_y = 0
    
    # Make the code work with a scalar.
    if np.isscalar(epsilon):
        eps_x = epsilon
        eps_y = epsilon
    
    # If two values are given.
    if len(epsilon) == 2:
        eps_x = epsilon[0]
        eps_y = epsilon[1] 
    else:
        eps_x = 0
        eps_y = 0

    # If r3 is not given set it to be the same as the first circle
    if r3 is None:
        x3 = x1
        y3 = y1
        r3 = r1


    intercepts1 = intersection_of_circles(x1=x1, x2=x2, y1=y1, y2=y2, r1=r1, r2=r2)
    intercepts2 = intersection_of_circles(x1=x1, x2=x3, y1=y1, y2=y3, r1=r1, r2=r3)
    intercepts3 = intersection_of_circles(x1=x2, x2=x3, y1=y2, y2=y3, r1=r2, r2=r3)
    intercepts  = [*intercepts1, *intercepts2, *intercepts3]

    x_intercepts1 = circle_x_at_yeq(eps_y/2., xc=x1, yc=y1, r=r1)
    x_intercepts2 = circle_x_at_yeq(eps_y/2., xc=x2, yc=y2, r=r2)
    x_intercepts3 = circle_x_at_yeq(eps_y/2., xc=x3, yc=y3, r=r3)
    x_intercepts4 = circle_x_at_yeq(-eps_y/2., xc=x1, yc=y1, r=r1)
    x_intercepts5 = circle_x_at_yeq(-eps_y/2., xc=x2, yc=y2, r=r2)
    x_intercepts6 = circle_x_at_yeq(-eps_y/2., xc=x3, yc=y3, r=r3)
    x_intercepts  = [*x_intercepts1, *x_intercepts2, *x_intercepts3, 
                     *x_intercepts4 , *x_intercepts5, *x_intercepts6] 

    # the lower and upper bounds of the intersection area
    # defined as the maximum leftmose extent and the minimum rightmost extent
    xmax = np.min([x1 + np.abs(r1), x2 + np.abs(r2), x3 + np.abs(r3)])
    xmin = np.max([x1 - np.abs(r1), x2 - np.abs(r2), x3 - np.abs(r3)])

    if xmin >= xmax: 
        return np.array([0.,0.,0.,0.])

    # collect all the points. The points define the segments for piecewise integration.
    # Between any two points, make sure no two circles cross and that no circle crosses
    # the x-axis.
    #
    # This means the integral of the lowest(highest) curve will alwasy be the 
    # lowest(highest) magnitude on any given segment.
    xpoints = [xmin, *intercepts, xmax, 0., *x_intercepts, (-eps_x/2.), (eps_x/2.)]
    
    

    # Check if the points fall in the domain. 
    combined_flag = np.isfinite(xpoints) & (xpoints >= xmin) & (xpoints <= xmax)

    flag_count = np.sum(combined_flag)

    if flag_count < 2: 
        return np.array([0.,0.,0.,0.])

    xi = np.array(xpoints)[combined_flag]
    xi = np.array(sorted(xi))

    rhs = (xi >= eps_x/2.)     #np.where(xi >= (eps_x/2.))
    lhs = (xi <= (-eps_x/2.))  #np.where(xi <= (-eps_x/2.))

    nrhs = np.sum(rhs)
    nlhs = np.sum(lhs)

    area_pp = 0.
    area_mp = 0.
    area_pm = 0.
    area_mm = 0.
    area    = 0.

    if nrhs >= 2:
        low1 = np.zeros(nrhs - 1)
        low2 = np.zeros(nrhs - 1)
        low3 = np.zeros(nrhs - 1)

        high1 = np.zeros(nrhs - 1)
        high2 = np.zeros(nrhs - 1)
        high3 = np.zeros(nrhs - 1)

        gap = np.zeros(nrhs - 1)

        xip = xi[rhs]
        for i in range(0, nrhs - 1):
            # lower half circle integrals
            low1[i] = int_lowersemicircle_dx(r=r1, xc=x1, yc=y1, x0=xip[i], x1=xip[i+1])
            low2[i] = int_lowersemicircle_dx(r=r2, xc=x2, yc=y2, x0=xip[i], x1=xip[i+1])
            low3[i] = int_lowersemicircle_dx(r=r3, xc=x3, yc=y3, x0=xip[i], x1=xip[i+1])

            # upper half circle integrals
            high1[i] = int_uppersemicircle_dx(r=r1, xc=x1, yc=y1, x0=xip[i], x1=xip[i+1])
            high2[i] = int_uppersemicircle_dx(r=r2, xc=x2, yc=y2, x0=xip[i], x1=xip[i+1])
            high3[i] = int_uppersemicircle_dx(r=r3, xc=x3, yc=y3, x0=xip[i], x1=xip[i+1])

            # gap integral
            gap[i] = (eps_y/2.)*(xip[i+1] - xip[i])

        low = np.max([low1, low2, low3], axis=0)
        high = np.min([high1, high2, high3], axis=0)

        if np.sum(np.isnan(low)) + np.sum(np.isnan(high)) > 0: sys.exit("Integral domain error!")

        dAp = np.maximum(np.maximum(high - gap, 0) - np.maximum((low - gap), 0), 0)
        dAm = np.maximum(np.minimum(high + gap, 0) - np.minimum((low + gap), 0), 0)

        dA = dAp + dAm


        area = area + np.sum(dA)
        area_pp = area_pp + np.sum(dAp)
        area_pm = area_pm + np.sum(dAm)

    
    if nlhs >= 2:
        low1 = np.zeros(nlhs - 1)
        low2 = np.zeros(nlhs - 1)
        low3 = np.zeros(nlhs - 1)

        high1 = np.zeros(nlhs - 1)
        high2 = np.zeros(nlhs - 1)
        high3 = np.zeros(nlhs - 1)

        gap = np.zeros(nlhs - 1)

        xim = xi[lhs]

        for i in range(0, nlhs - 1):
            # lower half circle integrals
            low1[i] = int_lowersemicircle_dx(r=r1, xc=x1, yc=y1, x0=xim[i], x1=xim[i+1])
            low2[i] = int_lowersemicircle_dx(r=r2, xc=x2, yc=y2, x0=xim[i], x1=xim[i+1])
            low3[i] = int_lowersemicircle_dx(r=r3, xc=x3, yc=y3, x0=xim[i], x1=xim[i+1])

            # upper half circle integrals
            high1[i] = int_uppersemicircle_dx(r=r1, xc=x1, yc=y1, x0=xim[i], x1=xim[i+1])
            high2[i] = int_uppersemicircle_dx(r=r2, xc=x2, yc=y2, x0=xim[i], x1=xim[i+1])
            high3[i] = int_uppersemicircle_dx(r=r3, xc=x3, yc=y3, x0=xim[i], x1=xim[i+1])

            # gap integral
            gap[i] = (eps_y/2.)*(xim[i+1] - xim[i])
        
        low = np.max([low1, low2, low3], axis=0)
        high = np.min([high1, high2, high3], axis=0)

        if np.sum(np.isnan(low)) + np.sum(np.isnan(high)) > 0: sys.exit("Integral domain error!")

        dAp = np.maximum(np.maximum(high - gap, 0) - np.maximum((low - gap), 0), 0)
        dAm = np.maximum(np.minimum(high + gap, 0) - np.minimum((low + gap), 0), 0)

        dA = dAp + dAm

        area = area + np.sum(dA)
        area_mp = area_mp + np.sum(dAp)
        area_mm = area_mm + np.sum(dAm)

    
    return np.array([area_pp, area_mp, area_pm, area_mm])

def transparency_factor(phi, theta):
    # for the transparancey model, we assume that the eight grids are
    # clocked evenly with respect to the flow angle into the cup.


    tanalpha = np.sqrt(np.tan(np.radians(phi))**2 + np.tan(np.radians(theta))**2)
    grid_thick = 0.1                 # vertical thickness of grid
    grid_cell  = 1.                  # mm**2, actual
    tau_0      = 0.888               # transparent fraction
    trans_area = tau_0*grid_cell     # transparent area of 1 cell
    l          = np.sqrt(trans_area) # lengthscale
    ngrids     = 8
    d          = (grid_thick/l)*tanalpha

    clockangles =  np.arange(ngrids)/90.

    # This was done explicitly in IDL does this work
    t = tau_0*(1.0 - np.outer(d,np.sin(np.radians(clockangles)))*
              (1.0 - np.outer(d,np.cos(np.radians(clockangles)))))

    # Therefore we should have t0 to t7. 
    return np.prod(t, axis=1)   # This will return an array of len(theta)

def psp_swp_spc_coldspot(phi, theta):
    
    if np.isscalar(phi) and np.isscalar(theta):
        phi = np.array([phi])
        theta = np.array([theta])

    # circle 1
    r1 = glbl_param_dict['re']
    x1 = glbl_param_dict['ze']*np.tan(np.radians(phi))
    y1 = glbl_param_dict['ze']*np.tan(np.radians(theta))

    # circle 2
    r2 = glbl_param_dict['rl']
    x2 = glbl_param_dict['zl']*np.tan(np.radians(phi))
    y2 = glbl_param_dict['zl']*np.tan(np.radians(theta))

    # circle 3
    r3 = glbl_param_dict['rc']
    x3 = 0.
    y3 = 0.

    eps_x = np.maximum(glbl_param_dict['epsilon'] - glbl_param_dict['dz_coll'] * np.tan(np.radians(np.abs(phi))), 0)
    eps_y = np.maximum(glbl_param_dict['epsilon'] - glbl_param_dict['dz_coll'] * np.tan(np.radians(np.abs(theta))), 0)

    areas = np.zeros([4, len(x1)])
    trasp = transparency_factor(phi, theta)
    for i in range(0, len(x1)):
        val        = quadrant_area_of_intersection(x1=x1[i], 
                     x2=x2[i], y1=y1[i], y2=y2[i], r1=r1,
                     r2=r2, x3=x3, y3=y3, r3=r3,
                     epsilon=[eps_x[i], eps_y[i]])

        areas[:,i] = trasp[i]*val

    return areas

