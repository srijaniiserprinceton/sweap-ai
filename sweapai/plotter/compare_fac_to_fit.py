import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import matplotlib.colors as mcolors

from sweapai.bimax_fitter_3D.fit_bimax import (single_bimax, bimax_coreandbeam)
from sweapai.src import span_functions

def bimax_coreandbeam_FAC(params, vgrid_FAC, bhat, Udrift_sign, component='coreandbeam'):
    logA_CORE, Ux_CORE, Uy_CORE, Uz_CORE, log_vth_para_CORE, log_vth_perp_CORE = params[:6]
    logA_BEAM, Udrift_BEAM, log_vth_para_BEAM, log_vth_perp_BEAM = params[6:]

    # adding in the sign of the beam drift
    # Udrift_BEAM = Udrift_sign * Udrift_BEAM

    # making the model parameters for the bi-Max distribution
    A_CORE = 10**logA_CORE
    oneover_vth_para_CORE = 10**(-log_vth_para_CORE)
    oneover_vth_perp_CORE = 10**(-log_vth_perp_CORE)

    A_BEAM = 10**logA_BEAM
    oneover_vth_para_BEAM = 10**(-log_vth_para_BEAM)
    oneover_vth_perp_BEAM = 10**(-log_vth_perp_BEAM)

    # computing a single bi-Max distribution for the core
    vpara_minus_U_CORE = vgrid_FAC[0]
    vpara_minus_U_BEAM = vgrid_FAC[0] + Udrift_BEAM       #Udrift_sign * Udrift_BEAM
    vperp = vgrid_FAC[1]

    exponent_CORE = -0.5 * ((vpara_minus_U_CORE * oneover_vth_para_CORE)**2 + (vperp * oneover_vth_perp_CORE)**2)
    exponent_BEAM = -0.5 * ((vpara_minus_U_BEAM * oneover_vth_para_BEAM)**2 + (vperp * oneover_vth_perp_BEAM)**2)

    return A_CORE * np.exp(exponent_CORE) + A_BEAM * np.exp(exponent_BEAM)  

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

    # making the denser grid for smoother plotting
    vpara_dense = np.linspace(np.min(vpara), np.max(vpara), 20)
    vperp_dense = np.linspace(np.min(vperp), np.max(vperp), 20)
    vvpara_dense, vvperp_dense = np.meshgrid(vpara_dense, vperp_dense, indexing='ij')
    vgrid_dense = np.array([vvpara_dense.flatten(), vvperp_dense.flatten()])

    if(component == 'coreandbeam'):
        model_vdf_dense = vdfdata_max * bimax_coreandbeam_FAC(fit_result.x, vgrid_dense, bhat, Udrift_sign, component='coreandbeam')

    # plot the original SPAN-L2 data and the fitted model
    levels = np.logspace(np.log10(vdfdata_max)-6,
                         np.log10(vdfdata_max), 20)

    if(component == 'coreonly'):
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
        plt.title('Residual VDF')
        norm = mcolors.TwoSlopeNorm(vcenter=0)

        # calculating the residual in log space
        residual = np.log10(vdfdata/vdfdata_max + floor) -\
                np.log10(model_vdf/vdfdata_max + floor)

        plt.tricontourf(vperp, vpara, residual,
                        cmap='seismic', norm=norm, levels=20)
        plt.colorbar(label='VDF Residual')
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.tight_layout()

    if(component == 'coreandbeam'):
        # plot the original SPAN-L2 data and the fitted model
        levels = np.log10(levels)
        plt.figure(figsize=(18, 10))
        plt.subplot(2, 2, 1)
        plt.title('Original SPAN-L2 VDF')
        plt.tricontourf(vperp, vpara, np.log10(vdfdata), levels=levels)
        plt.colorbar(label='VDF Value')
        plt.scatter(vperp, vpara, marker='x', color='k')
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.subplot(2, 2, 2)
        plt.title('SPAN-res Bi-Maxwellian VDF')
        plt.tricontourf(vperp, vpara, np.log10(model_vdf), levels=levels)
        plt.colorbar(label='VDF Value')
        plt.gca().set_aspect('equal', adjustable='box')


        plt.subplot(2, 2, 3)
        plt.title('High-res Bi-Maxwellian VDF')
        norm = mcolors.TwoSlopeNorm(vcenter=0)

        plt.tricontourf(vvperp_dense.flatten(), vvpara_dense.flatten(), np.log10(model_vdf_dense), levels=levels)
        plt.colorbar(label='VDF Residual')
        plt.gca().set_aspect('equal', adjustable='box')

        plt.subplot(2, 2, 4)
        plt.title('Residual VDF')
        norm = mcolors.TwoSlopeNorm(vcenter=0)

        # calculating the residual in log space
        residual = np.log10(vdfdata/vdfdata_max + floor) -\
                np.log10(model_vdf/vdfdata_max + floor)

        plt.tricontourf(vperp, vpara, residual,
                        cmap='seismic', norm=norm, levels=20)
        plt.colorbar(label='VDF Residual')
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.tight_layout()