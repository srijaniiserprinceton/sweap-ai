import numpy as np
from scipy.optimize import least_squares

NAX = np.newaxis

def clean_and_scale(vdfdata, vgrid):
    vdfdata = np.asarray(vdfdata, dtype=float).ravel()

    good = (
        np.isfinite(vdfdata)
        & (vdfdata > 0)
    )

    vdfdata = vdfdata[good]
    vgrid = vgrid[:,good]

    # should break if there are bad values
    # which should have been filtered out
    vdfdata_max = np.max(vdfdata[vdfdata > 0])
    vdfdata_scaled = vdfdata / vdfdata_max

    return vdfdata_scaled, vgrid, vdfdata_max

def single_bimax(params, vgrid, bhat):
    logA, Ux, Uy, Uz, log_vth_para, log_vth_perp = params

    # making the model parameters for the bi-Max distribution
    A = 10**logA
    oneover_vth_para = 10**(-log_vth_para)
    oneover_vth_perp = 10**(-log_vth_perp)

    # forming the covariance matrix containing the thermal speeds
    bbT = np.outer(bhat, bhat)
    I = np.eye(3)
    oneover_Sigma = oneover_vth_para**2 * bbT + oneover_vth_perp**2 * (I - bbT)

    # computing the single bi-Max distribution
    vgrid_minus_U = vgrid - np.array([Ux, Uy, Uz])[:,NAX]

    exponent = -0.5 * np.einsum('ji,jk,ki->i', vgrid_minus_U, oneover_Sigma, vgrid_minus_U)
    return A * np.exp(exponent)

def bimax_coreandbeam(params, vgrid, bhat, Udrift_sign):
    logA_CORE, Ux_CORE, Uy_CORE, Uz_CORE, log_vth_para_CORE, log_vth_perp_CORE = params[:6]
    logA_BEAM, Udrift_BEAM, log_vth_para_BEAM, log_vth_perp_BEAM = params[6:]

    # adding in the sign of the beam drift
    Udrift_BEAM = Udrift_sign * Udrift_BEAM

    # making the model parameters for the bi-Max distribution
    A_CORE = 10**logA_CORE
    oneover_vth_para_CORE = 10**(-log_vth_para_CORE)
    oneover_vth_perp_CORE = 10**(-log_vth_perp_CORE)

    A_BEAM = 10**logA_BEAM
    oneover_vth_para_BEAM = 10**(-log_vth_para_BEAM)
    oneover_vth_perp_BEAM = 10**(-log_vth_perp_BEAM)

    # forming the covariance matrix containing the thermal speeds
    bbT = np.outer(bhat, bhat)
    I = np.eye(3)
    oneover_Sigma_CORE = oneover_vth_para_CORE**2 * bbT + oneover_vth_perp_CORE**2 * (I - bbT)
    oneover_Sigma_BEAM = oneover_vth_para_BEAM**2 * bbT + oneover_vth_perp_BEAM**2 * (I - bbT)

    # computing a single bi-Max distribution for the core
    vgrid_minus_U_CORE = vgrid - np.array([Ux_CORE, Uy_CORE, Uz_CORE])[:,NAX]

    # computing a single bi-Max distribution for the beam
    U_BEAM = np.array([Ux_CORE + Udrift_BEAM * bhat[0], Uy_CORE + Udrift_BEAM * bhat[1], Uz_CORE + Udrift_BEAM * bhat[2]])
    vgrid_minus_U_BEAM = vgrid - U_BEAM[:,NAX]

    exponent_CORE = -0.5 * np.einsum('ji,jk,ki->i', vgrid_minus_U_CORE, oneover_Sigma_CORE, vgrid_minus_U_CORE)
    exponent_BEAM = -0.5 * np.einsum('ji,jk,ki->i', vgrid_minus_U_BEAM, oneover_Sigma_BEAM, vgrid_minus_U_BEAM)

    return A_CORE * np.exp(exponent_CORE) + A_BEAM * np.exp(exponent_BEAM)  

def core_fitting_only(tidx, vgrid, vdfdata, bhat, span_L3):
    # scale the vdf
    vdfdata, vgrid, vdfdata_max = clean_and_scale(vdfdata, vgrid)

    # creating initial guess for parameters
    logA_init = 0.0
    U_init = span_L3.VEL_INST[tidx].data
    log_vth_para_init = np.log10(200)
    log_vth_perp_init = np.log10(200)

    params = np.array([logA_init,
                       U_init[0], U_init[1], U_init[2],
                       log_vth_para_init, log_vth_perp_init])

    def residual_linear(params):
        model_vdf = single_bimax(params, vgrid, bhat)

        return model_vdf - vdfdata
    
    def residual_log(params):
        model_vdf = single_bimax(params, vgrid, bhat)

        return np.log10(model_vdf + 1e-12) - np.log10(vdfdata + 1e-12)

    def objective(params):
        r = residual_linear(params)
        return np.sum(r**2)

    # performing the least-squares optimization
    lsq = least_squares(residual_linear, params, 
                        method='trf', verbose=2)

    return lsq, vdfdata_max


def core_and_beam_fitting(tidx, vgrid, vpara, vdfdata, bhat, fit_coreonly):
    # scale the vdf
    vdfdata, vgrid, vdfdata_max = clean_and_scale(vdfdata, vgrid)

    # finding the maximum distance of the domain in the vpara direction from the core vpara
    Dvpara_domain = np.max(np.abs(vpara))

    # creating initial guess for parameters: CORE
    logA_init_CORE = fit_coreonly.x[0]
    U_init_CORE = fit_coreonly.x[1:4]
    log_vth_para_init_CORE = fit_coreonly.x[4]
    log_vth_perp_init_CORE = fit_coreonly.x[5]

    params_core = np.array([logA_init_CORE,
                            U_init_CORE[0], U_init_CORE[1], U_init_CORE[2],
                            log_vth_para_init_CORE, log_vth_perp_init_CORE])

    # creating bounds for parameters: CORE
    logA_bounds_CORE = (logA_init_CORE - 1, logA_init_CORE + 1)
    U_bounds_CORE = (U_init_CORE - 100, U_init_CORE + 100)
    log_vth_para_bounds_CORE = (log_vth_para_init_CORE - 1, log_vth_para_init_CORE + 1)
    log_vth_perp_bounds_CORE = (log_vth_perp_init_CORE - 1, log_vth_perp_init_CORE + 1)

    bounds_CORE = (np.array([logA_bounds_CORE[0], U_bounds_CORE[0][0], U_bounds_CORE[0][1], U_bounds_CORE[0][2], log_vth_para_bounds_CORE[0], log_vth_perp_bounds_CORE[0]]),
                   np.array([logA_bounds_CORE[1], U_bounds_CORE[1][0], U_bounds_CORE[1][1], U_bounds_CORE[1][2], log_vth_para_bounds_CORE[1], log_vth_perp_bounds_CORE[1]]))

    # creating initial guess for parameters: BEAM
    logA_init_BEAM = logA_init_CORE - 2
    Udrift_init_BEAM = (10**log_vth_para_init_CORE) * 2
    log_vth_para_init_BEAM = fit_coreonly.x[4] * 1.0
    log_vth_perp_init_BEAM = fit_coreonly.x[5] * 1.0

    params_beam = np.array([logA_init_BEAM, Udrift_init_BEAM,
                            log_vth_para_init_BEAM, log_vth_perp_init_BEAM])

    # creating bounds for parameters: BEAM
    logA_bounds_BEAM = (logA_init_CORE - 6, logA_init_CORE)
    Udrift_bounds_BEAM = (10**log_vth_para_init_CORE, Dvpara_domain)
    log_vth_para_bounds_BEAM = (-np.inf, np.log10(Dvpara_domain))
    log_vth_perp_bounds_BEAM = (-np.inf, np.log10(Dvpara_domain))

    bounds_BEAM = (np.array([logA_bounds_BEAM[0], Udrift_bounds_BEAM[0], log_vth_para_bounds_BEAM[0], log_vth_perp_bounds_BEAM[0]]),
                   np.array([logA_bounds_BEAM[1], Udrift_bounds_BEAM[1], log_vth_para_bounds_BEAM[1], log_vth_perp_bounds_BEAM[1]]))


    # making combined params and bounds
    params = np.concatenate([params_core, params_beam])

    bounds = (np.concatenate([bounds_CORE[0], bounds_BEAM[0]]),
              np.concatenate([bounds_CORE[1], bounds_BEAM[1]]))

    # beam drift sign
    Udrift_sign = np.sign(fit_coreonly.x[1:4] @ bhat)

    def residual_log(params):
        model_vdf = bimax_coreandbeam(params, vgrid, bhat, Udrift_sign)

        return np.log10(model_vdf + 1e-12) - np.log10(vdfdata + 1e-12)

    # performing the least-squares optimization
    lsq = least_squares(residual_log, params, bounds=bounds, loss='linear',
                        method='trf', verbose=2)

    return lsq, vdfdata_max