import jax.numpy as jnp
import numpy as np
from line_profiler import profile

from sweapai.src import functions as fn

NAX = np.newaxis

def SPANpolar_to_SPANcartesian(span_L2):
    energy = span_L2.energy.data * 1.0
    theta = span_L2.theta.data * 1.0
    phi = span_L2.phi.data * 1.0

    m_p = 0.010438870    # eV/c^2 where c = 299792 km/s
    q_p = 1

    velocity = np.sqrt(2 * q_p * energy / m_p)

    # Define the Cartesian Coordinates
    vx = velocity * np.cos(np.radians(theta)) * np.cos(np.radians(phi))
    vy = velocity * np.cos(np.radians(theta)) * np.sin(np.radians(phi))
    vz = velocity * np.sin(np.radians(theta))

    # making sure that the time axis is the 0th dimension
    return np.swapaxes(np.array([vx, vy, vz]), 0, 1)

@profile
def project_SPANgrids_to_FAgrids(biMax, spangrids, bvec):
    # shifting from the instrument frame to the plasma frame
    PFgrids = spangrids - biMax['v_core'][:,NAX,NAX,NAX]

    # taking projection along the magnetic field direction
    vpara = PFgrids[0] * bvec[0] + PFgrids[1] * bvec[1] + PFgrids[2] * bvec[2]

    # computing the perpendicular component
    vsq = PFgrids[0]**2 + PFgrids[1]**2 + PFgrids[2]**2
    vperp = np.sqrt(vsq - vpara**2)

    return (vpara, vperp)

@profile
def rotate_SPANgrids_to_FAgrids(biMax, spangrids, bvec):
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
    PFgrids = spangrids - biMax['v_core'][:,NAX,NAX,NAX]

    # rotating to align with the magnetic field direction
    FAgrids = fn.rotate_vector_field_aligned(*PFgrids, 
                                             *fn.field_aligned_coordinates(bvec))

    vpara = FAgrids[0]
    vperp = np.sqrt(FAgrids[1]**2 + FAgrids[2]**2)

    return (vpara, vperp)