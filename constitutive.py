import numpy as np
from scipy import special as sp



def mat_to_py(var):
    # converts matlab arrays to numpy arrays and makes sure everything is shape Nx1
    val = np.array(var)
    if len(val.shape) < 2:
        val = val[:,np.newaxis]
        
    L, W = val.shape

    if L < W:
        val = val.T
    return val



def viscosity(h2o_d, phi_s_eta, gdot, m):
    """
    Calculates viscosity of silicate melt using 
    - dacite melt viscosity model from Whittington 2009
    - crystal content relative viscosity model from Costa 2005
    
    h2o_d in wt%
    phi_s_eta = phi_s / (phi_s + phi_l)
    gdot is strain rate
    """
    
    h2o_arg = np.log10(h2o_d + 0.26)
    logeta_m = -4.43 + (7618.3 - 17.25*h2o_arg)/(m['T'] - 406.1 + (292.6*h2o_arg))
    eta_m = np.power(10,logeta_m)
    
    if gdot == 0:
        #values from Costa 2005b and 2007b
        kappa   = 0.999916
        phistar = 0.673
        gamma   = 3.98937
        delta   = 16.9386
    else:
        # strain-rate dependent values from Costa 2005 and Caricchi 2007
        phistar = -0.066499*np.tanh(0.913424*np.log10(gdot)+3.850623) + 0.591806
        delta   = -6.301095*np.tanh(0.818496*np.log10(gdot)+2.86)     + 7.462405
        kappa   = -0.000378*np.tanh(1.148101*np.log10(gdot)+3.92)     + 0.999572
        gamma   =  3.987815*np.tanh(0.890800*np.log10(gdot)+3.24)     + 5.099645
        
    pih = np.sqrt(np.pi)
    B   = 2.5
    
    phifr = phi_s_eta/phistar
    
    top = (1. + np.power(phifr,delta))    #numerator
    efarg = 0.5*pih*phifr/kappa*(1. + np.power(phifr,gamma))  #error function argument
    eta_phi = top/np.power((1. - kappa*sp.erf(efarg)), B*phistar)

    eta = eta_m*eta_phi
        
    return eta, eta_m, eta_phi


def density (p, phi_g, mh, m):
    """
    calculate magma density given pressure, porosity and mole fraction water
    """
    
    # get dissolved volatiles
    h2o_d, co2_d, dh2o_d, dco2_d = solubility(p, mh, m)
    
    c1  = 1/(1 - h2o_d - co2_d)
    c2  = 1/(1 + h2o_d*c1*(m['rho_l']/m['rho_hd']) + co2_d*c1*(m['rho_l']/m['rho_cd']))
    c12 = c1*c2
    
    chi_s, dp, cf0, cf1, cf2 = solid_mass_frac(p*1e-6, m['pf'])
    
    phi_s = chi_s*m['rho_l']*c12*(1-phi_g)/(m['rho_s'] + chi_s*(c12*m['rho_l']-m['rho_s']))
    phi_s_eta = phi_s/(1-phi_g)
    
    phi_l = (1 - phi_s - phi_g)*c2 
    rho_g = p*(mh/(m['Rw']*m['T']) + (1-mh)/(m['Rc']*m['T']))
    
    rho = m['rho_s']*phi_s + m['rho_l']*phi_l*c1 + rho_g*phi_g
    
    return rho, c1, c2, phi_s, phi_l, rho_g
    


def solubility(p, mh, m):
    """
    calculate dissolved h2o and co2 using pressure, temperature, mole fraction water
    using constitutive relations from Liu et al 2005
    Be careful with units!
    
    p    pressure in Pa
    mh   mole fraction water (range 0 to 1)
    m    dictionary of values
    
    h2o_d   dissolved h2o in wt%
    co2_d   dissolved co2 in ppm
    
    """
    
    # convert p to MPa
    p = 1e-6*p
    
    T = m['T']
    
    # Precompute for speed.
    Ph      = p*mh
    Pc      = p*(1-mh)
    sqrtPh  = np.sqrt(Ph)
    Ph15    = Ph*sqrtPh # Ph^1.5
    sqrtPhneg = 1/sqrtPh
    
    # These equations require MPa, Kelvin, and return wt% and ppm in weight. Assuming saturation!
    #h2o_d = (354.94*sqrtPw + 9.623*Pw - 1.5223*Pw15)/T + 0.0012439*Pw15 + Pc*(-1.084e-4*sqrtPw - 1.362e-5*Pw)
    #co2_d = Pc*(5668 - 55.99*Pw)/T + Pc*(0.4133*sqrtPw + 2.041e-3*Pw15)
    
    # constants in equations
    a1 =  354.94
    a2 =  9.623
    a3 = -1.5223
    a4 =  0.0012439
    a5 = -1.084e-4
    a6 = -1.362e-5

    b1 =  5668
    b2 = -55.99
    b3 =  0.4133
    b4 =  2.041e-3

    # These equations require MPa, Kelvin, and return wt% and ppm in weight.
    # Assuming saturation!
    h2o_d = (a1*sqrtPh + a2*Ph + a3*Ph15)/T + a4*Ph15 + Pc*(a5*sqrtPh + a6*Ph)
    co2_d = Pc*(b1 + b2*Ph)/T + Pc*(b3*sqrtPh + b4*Ph15)
    
    # convert to mass fractions
    h2o_d = 1e-2*h2o_d
    co2_d = 1e-6*co2_d
    
    # solubility derivatives
    # NB: there could easily be a typo here. CHECK AND CHECK AGAIN!!
    dh2odph = (a1*0.5*sqrtPhneg + a2 + a3*1.5*sqrtPh)/T + a4*1.5*sqrtPh + Pc*(a5*0.5*sqrtPhneg + a6)
    dh2odpc = (a5*sqrtPh + a6*Ph)

    dco2dph = Pc*b2/T + Pc*(b3*0.5*sqrtPhneg + b4*1.5*sqrtPh)
    dco2dpc = (b1 + b2*Ph)/T + (b3*sqrtPh + b4*Ph15)

    # convert to mass fraction
    dh2o_d       = {}
    dh2o_d['ph'] = 1e-2*dh2odph
    dh2o_d['pc'] = 1e-2*dh2odpc
    
    dco2_d       = {}
    dco2_d['ph'] = 1e-6*dco2dph
    dco2_d['pc'] = 1e-6*dco2dpc
    
    return h2o_d, co2_d, dh2o_d, dco2_d
    
    
    
def solubility_deriv (p, mh, dh2o_d, dco2_d, m):
    """
    Derivatives for solubility mass fractions
    !!! Chain rule extravaganza !!!
    
    ph: partial pressure of water = mh x p
    pc: partial pressure of co2 = (1-mh) x p
    """
    
    dphdp  = 1e-6*mh
    dphdmh = 1e-6*p
    
    dpcdp  =  1e-6*(1-mh)
    dpcdmh = -1e-6*p
    
    dh2o_d['p']  = dh2o_d['ph']*dphdp  + dh2o_d['pc']*dpcdp
    dh2o_d['mh'] = dh2o_d['ph']*dphdmh + dh2o_d['pc']*dpcdmh
    
    dco2_d['p']  = dco2_d['ph']*dphdp  + dco2_d['pc']*dpcdp
    dco2_d['mh'] = dco2_d['ph']*dphdmh + dco2_d['pc']*dpcdmh
    
    return dh2o_d, dco2_d

    
    
def solid_mass_frac(pMPA, pp):
    """
    Calculate solid mass fraction for specified pressure using piecewise polynomials 
    Pressure in MPa here
    
    NB the shapes of these may become a problem. 
    NEEDS TO BE CHECKED!!!
    """    
    
    # check where the the pressure is between the two breakpoints
    ppbreak = pp['breaks'].transpose()
    tmp = np.all([pMPA>=ppbreak[0,:-1], pMPA<ppbreak[0,1:]], axis=0)
    
    # gather index of polynomial for each pressure in pMPA
    chi_ind = mat_to_py(np.argmax(tmp, axis=1))
    
    # make dp and coef matrices of size pMPA. Can't figure out how to make the coeffs in one line
    dp  = pMPA - pp['breaks'][chi_ind,0]
    cf0 = pp['coefs'][chi_ind,0]
    cf1 = pp['coefs'][chi_ind,1]
    cf2 = pp['coefs'][chi_ind,2]
    cf3 = pp['coefs'][chi_ind,3]
    
    # calculate piecewise polynomial
    chi_s = cf0*np.power(dp,3) + cf1*np.power(dp,2) + cf2*dp + cf3
    
    # check at low pressure, chi_s = 1
    chi_s[pMPA<pp['breaks'][ 0]] = pp['coefs'][0,3]
    
    # check that there are no negative values at extremely high pressures
    chi_s[chi_s<0] = 0
    
    return chi_s, dp, cf0, cf1, cf2



def solid_mass_frac_deriv (dp, cf0, cf1, cf2):
    """ calculate derivative for solid mass fraction chi_s, which only depends on pressure"""
    
    # additional 1e-6 to account for pressures in MPa in dp
    dchisdp = 1e-6*(3*cf0*np.power(dp,2) + 2*cf1*dp + cf2)
    
    return dchisdp


def magmaperm (phi_g, degas, m):
    """ Calculate magma permeability using Kozeny-Carman"""
    kmag = m['kc']*np.power(phi_g,3)*degas
    
    return kmag


def wallperm (z, m):
    """calculate wall rock permeability from Manning and Ingebritsen [1999]"""
    
    kwall = m['klw']['top'] / np.power(1e-3*np.absolute(z),m['klw']['mi'])
    
    return kwall


def lateralperm (phi_g, degas, z, m):
    """ Calculate lateral permeability, combination of wall rock and magma permeability"""
    
    kmag  = magmaperm(phi_g, degas, m)
    kwall = wallperm(z, m)
    klat  = (m['klw']['L']+m['R']) / (m['R']/kmag + m['klw']['L']/kwall)

    return klat


def c1c2_deriv (h2o_d, co2_d, c1, c2, dh2o_d, dco2_d, m):
    """
    calculate derivatives for c1, c2 needed for density equation
    !!! chain rule extravaganza !!!
    """
    
    dc1 = {}
    dc2 = {}
    
    # derivatives for c1
    c1_2      = c1*c1
    dc1['p']  = c1_2*(dh2o_d['p']  + dco2_d['p'] )
    dc1['mh'] = c1_2*(dh2o_d['mh'] + dco2_d['mh'])
    
    # derivatives for c2
    rholhd = m['rho_l']/m['rho_hd']
    rholcd = m['rho_l']/m['rho_cd']
        
    c2_2     = c2*c2
        
    dc2dh2od = -c2_2*(rholhd*(h2o_d*c1_2 + c1) + rholcd*co2_d*c1_2 )
    dc2dco2d = -c2_2*(rholhd*h2o_d*c1_2       + rholcd*(co2_d*c1_2 + c1))
    
    dc2['p']  = dc2dh2od*dh2o_d['p']  + dc2dco2d*dco2_d['p']
    dc2['mh'] = dc2dh2od*dh2o_d['mh'] + dc2dco2d*dco2_d['mh']
   
    return dc1, dc2, dc2dh2od, dc2dco2d


def phasefracs_deriv (phi_s, phi_g, chi_s, c1, c2, dchisdp, dc1, dc2, m):
    """
    derivatives of phase fractions
    !!! chain rule extravaganza !!!
    """
    
    c12   = c1*c2
    rhosl = m['rho_s']*m['rho_l']
    
    denom  = chi_s*(m['rho_l']*c12 - m['rho_s']) + m['rho_s']
    denom2 = np.power(denom,2)
    
    dphisdchi = c12*rhosl*(1-phi_g)/denom2
    dphisdc1  =  c2*rhosl*(1-phi_g)*(1-chi_s)*chi_s/denom2
    dphisdc2  =  c1*rhosl*(1-phi_g)*(1-chi_s)*chi_s/denom2
    
    # solid volume fraction derivs
    dphis          = {}
    dphis['p']     = dphisdchi*dchisdp + dphisdc1*dc1['p'] + dphisdc2*dc2['p']
    dphis['phi_g'] = -c12*m['rho_l']*chi_s/denom
    dphis['mh']    = dphisdc1*dc1['mh'] + dphisdc2*dc2['mh']
    
    # liquid volume fraction derivs
    dphil          = {}
    dphil['p']     = -c2*dphis['p'] + (1 - phi_s - phi_g)*dc2['p']
    dphil['phi_g'] =  c2*(-dphis['phi_g'] - 1)
    dphil['mh']    = -c2*dphis['mh'] + (1 - phi_s - phi_g)*dc2['mh']
    
    return dphis, dphil

