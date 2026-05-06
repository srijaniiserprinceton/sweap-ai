import jax.numpy as jnp

import functions as fn

def SPANgrids_to_FAgrids(biMax, spangrids, bvec):
    '''
    This function takes the SPAN grids and performs the forward 
    transformation to the FA grids. The transformation consists of two steps:
    1. shifting from the instrument frame to the plasma frame
    2. rotating to align with the magnetic field direction

    Parameters
    ----------
    biMax : dict
        A dictionary containing the bi-Maxwellian parameters in the spacecraft frame(?).
        (n, T_perp, T_paral, v_drift) for each population.
    spangrids : array-like
        An array containing the SPAN grids [Vx, Vy, Vz] for each measurement.
    bvec : array-like
        The magnetic field vector for that measurement.
    '''
    # shifting from the instrument frame to the plasma frame
    PFgrids = spangrids - biMax['v_core']

    # rotating to align with the magnetic field direction
    FAgrids = fn.inverse_rotate_vector_field_aligned(*PFgrids, 
                                                     *fn.field_aligned_coordinates(bvec))

    return FAgrids