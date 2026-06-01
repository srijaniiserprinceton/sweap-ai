import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import matplotlib.colors as mcolors

from sweapai.bimax_fitter_3D.fit_bimax import (single_bimax, bimax_coreandbeam)
from sweapai.src import span_functions

def plot_comparison(vdfdata, vdfdata_max, vgrid, bhat, fit_result, component='coreonly', floor=1e-18):
    # extract the fitted parameters
    Ux_fit, Uy_fit, Uz_fit = fit_result.x[1:4]

    # beam drift sign
    Udrift_sign = np.sign(np.array([Ux_fit, Uy_fit, Uz_fit]) @ bhat)

    # computing the FAC coordinates for the vgrid
    vpara, vperp = span_functions.project_SPANgrids_to_FAgrids(
                    biMax={'v_core': np.array([Ux_fit, Uy_fit, Uz_fit])},
                    spangrids=vgrid,
                    bvec=bhat
                    )

    # compute the fitted bi-Maxwellian distribution
    if(component == 'coreonly'):
        model_vdf = vdfdata_max * single_bimax(fit_result.x, vgrid, bhat)
    elif(component == 'coreandbeam'):
        model_vdf = vdfdata_max * bimax_coreandbeam(fit_result.x, vgrid, bhat, Udrift_sign)

    # making the reflected side for plotting
    vpara = np.concatenate([vpara, vpara])
    vperp = np.concatenate([vperp, -vperp])
    vdfdata = np.concatenate([vdfdata, vdfdata])
    model_vdf = np.concatenate([model_vdf, model_vdf])

    # plot the original SPAN-L2 data and the fitted model
    levels = np.logspace(np.log10(vdfdata_max)-6,
                         np.log10(vdfdata_max), 20)

    '''
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original SPAN-L2 VDF')
    plt.tricontourf(vperp, vpara, vdfdata, levels=levels)
    plt.colorbar(label='VDF Value')
    
    plt.subplot(1, 2, 2)
    plt.title('Fitted Bi-Maxwellian VDF')
    plt.tricontourf(vperp, vpara, model_vdf, levels=levels)
    plt.colorbar(label='VDF Value')
    
    plt.tight_layout()
    plt.show()
    '''

    # plot the original SPAN-L2 data and the fitted model
    levels = np.log10(levels)
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.title('Original SPAN-L2 VDF')
    plt.tricontourf(vperp, vpara, np.log10(vdfdata), levels=levels)
    plt.colorbar(label='VDF Value')
    plt.scatter(vperp, vpara, marker='x', color='k')
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.subplot(1, 3, 2)
    plt.title('Fitted Bi-Maxwellian VDF')
    plt.tricontourf(vperp, vpara, np.log10(model_vdf), levels=levels)
    plt.colorbar(label='VDF Value')
    plt.gca().set_aspect('equal', adjustable='box')

    plt.subplot(1, 3, 3)
    plt.title('Fitted Bi-Maxwellian VDF')
    norm = mcolors.TwoSlopeNorm(vcenter=0)

    # calculating the residual in log space
    residual = np.log10(vdfdata/vdfdata_max + floor) -\
               np.log10(model_vdf/vdfdata_max + floor)

    plt.tricontourf(vperp, vpara, residual,
                    cmap='seismic', norm=norm, levels=20)
    plt.colorbar(label='VDF Residual')
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.tight_layout()