def SPCgrids_to_FAgrids(biMax, spcgrids, bvec):
    '''
    This function takes the SPAN grids and performs the forward 
    transformation to the FA grids. The transformation consists of two steps:
    1. shifting from the instrument frame to the plasma frame
    2. rotating to align with the magnetic field direction

    Parameters
    ----------
    biMax : dict
        A dictionary containing the bi-Maxwellian parameters 
        (n, T_perp, T_paral, v_drift) for each population.
    spcgrids : array-like
        An array containing the SPC grids [Vx (fid), Vy (fid), Vz (true)] for each measurement.
    bvec : array-like
        The magnetic field vector for that measurement.
    '''
    # shifting from the instrument frame to the plasma frame
    PFgrids = spcgrids - biMax['v_core']

    # rotating to align with the magnetic field direction
    FAgrids = fn.inverse_rotate_vector_field_aligned(*PFgrids, 
                                                     *fn.field_aligned_coordinates(bvec))

    return FAgrids