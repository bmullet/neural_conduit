import numpy as np
from scipy import special as sp
from scipy.optimize import fsolve
import utils
import copy
import tensorflow as tf




def reformat_params (m):
    """
    output m dictionary from smf or tdc is a bit tedious for what we want here. 
    Need to clean up a few things to make sure that m is suitable for use in ssc in domegoveqn
    
    Input : m structure from ss['m']
    Output: slightly reformatted dictionary of m for use in ssc in domegoveqn
    """
    
    mout = copy.deepcopy(m)
    
    mout.pop('pf', None)
    mout.pop('chi_ch', None)
    
    mout['pf'] = {}
    mout['pf']['breaks'] = utils.mat_to_py(m['pf']['pp']['breaks'])
    mout['pf']['coefs']  = utils.mat_to_py(m['pf']['pp']['coefs'])
    
    mout['chi_ch'] = {'h2o':m['chi_ch']['total']['h2o'], 'co2':m['chi_ch']['total']['co2']}
    
    mout['kc'] = m['k_lat']
    mout['fr']['A'] = 2*m['fr']['v_ref']*np.exp(-m['fr']['f0']/m['fr']['a'])   # prefactor in MBE

    return mout


def viscosity(h2o_d, phi_s_eta, gdot, m):
    """
    Calculates viscosity of silicate melt using 
    - dacite melt viscosity model from Whittington 2009
    - crystal content relative viscosity model from Costa 2005
    
    h2o_d in wt%
    phi_s_eta = phi_s / (phi_s + phi_l)
    gdot is strain rate
    """
    
    eta = {}
    
    h2o_arg = np.log10(h2o_d + 0.26)
    logeta_m = -4.43 + (7618.3 - 17.25*h2o_arg)/(m['T'] - 406.1 + (292.6*h2o_arg))
    eta['melt'] = np.power(10,logeta_m)
    
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
    eta['xtal'] = top/np.power((1. - kappa*sp.erf(efarg)), B*phistar)

    eta['mix'] = eta['melt']*eta['xtal']
        
    return eta


def density (p, phi_g, mh, m):
    """
    calculate magma density given pressure, porosity and mole fraction water
    """
    
    # get dissolved volatiles
    h2o, co2 = solubility(p, mh, m)
    
    c       = {}
    c['c1'] = 1/(1 - h2o['dissolved'] - co2['dissolved'])
    c['c2'] = 1/(1 + h2o['dissolved']*c['c1']*(m['rho_l']/m['rho_hd']) + co2['dissolved']*c['c1']*(m['rho_l']/m['rho_cd']))
    c12     = c['c1']*c['c2']
    c['dc1'], c['dc2'] = c1c2_deriv(h2o, co2, c, m)
    
    # solid mass, volume fractions
    phi = {}
    phi['chi_s'] = solid_mass_frac(p*1e-6, m['pf'])
    phi['chi_s']['dchisdp'] = solid_mass_frac_deriv(phi['chi_s'])
    
    chi_s    = phi['chi_s']['chi_s']
    phi['s'] = chi_s*m['rho_l']*c12*(1-phi_g)/(m['rho_s'] + chi_s*(c12*m['rho_l']-m['rho_s']))
    phi['l'] = (1 - phi['s'] - phi_g)*c['c2'] 
    phi['dphis'], phi['dphil'] = phasefracs_deriv(phi_g, phi, phi['chi_s'], c, m)
    
    rho = {}
    rho['g']   = p*(mh/(m['Rw']*m['T']) + (1-mh)/(m['Rc']*m['T']))
    rho['mix'] = m['rho_s']*phi['s'] + m['rho_l']*phi['l'] *c['c1'] + rho['g']*phi_g
    rho['drhog'] = gasdensity_deriv(p, mh, m)
    
    return h2o, co2, c, phi, rho
    

def exsolvedco2h2o(mh, m):
    
    gamma        = {}
    gamma['g']   = (1 - mh)/(m['B']*mh)
    
    # precalculate gamma factors for h and c 
    gamma['h']   = 1/(1+gamma['g'])
    gamma['c']   = gamma['g']/(1+gamma['g'])
    
    gamma['dmh'] = 1/m['B']/np.power(1+gamma['g'],2)/np.power(mh,2)
    # ^^ this is d/dmh of 1/(1+Gamma).   d/dmh of Gamma/(1+Gamma) is the negative of this
    
    return gamma
    

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
    
    # outputs
    h2o = {}
    co2 = {}
    
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
    
    # solubility derivatives
    # NB: there could easily be a typo here. CHECK AND CHECK AGAIN!!
    dh2odph = (a1*0.5*sqrtPhneg + a2 + a3*1.5*sqrtPh)/T + a4*1.5*sqrtPh + Pc*(a5*0.5*sqrtPhneg + a6)
    dh2odpc = (a5*sqrtPh + a6*Ph)

    dco2dph = Pc*b2/T + Pc*(b3*0.5*sqrtPhneg + b4*1.5*sqrtPh)
    dco2dpc = (b1 + b2*Ph)/T + (b3*sqrtPh + b4*Ph15)
        
    # convert to mass fractions for outputs. 
    # NB this is different from tdcFV where the output is in wt% and ppm
    h2o['dissolved'] = 1e-2*h2o_d
    co2['dissolved'] = 1e-6*co2_d
    
    h2o['dph'] = 1e-2*dh2odph
    h2o['dpc'] = 1e-2*dh2odpc
    
    co2['dph'] = 1e-6*dco2dph
    co2['dpc'] = 1e-6*dco2dpc

    h2o, co2 = solubility_deriv(1e6*p, mh, h2o, co2, m)
    
    return h2o, co2
    
    
    
def solubility_deriv (p, mh, h2o, co2, m):
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
    
    h2o['dp']  = h2o['dph']*dphdp  + h2o['dpc']*dpcdp
    h2o['dmh'] = h2o['dph']*dphdmh + h2o['dpc']*dpcdmh
    
    co2['dp']  = co2['dph']*dphdp  + co2['dpc']*dpcdp
    co2['dmh'] = co2['dph']*dphdmh + co2['dpc']*dpcdmh
    
    return h2o, co2

    
    
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
    chi_ind = utils.mat_to_py(np.argmax(tmp, axis=1))
    
    # make dp and coef matrices of size pMPA. Can't figure out how to make the coeffs in one line
    dp  = pMPA - pp['breaks'][chi_ind,0]
    cf0 = pp['coefs'][chi_ind,0]
    cf1 = pp['coefs'][chi_ind,1]
    cf2 = pp['coefs'][chi_ind,2]
    cf3 = pp['coefs'][chi_ind,3]
    
    # calculate piecewise polynomial
    chi_s = cf0*np.power(dp,3) + cf1*np.power(dp,2) + cf2*dp + cf3
    
    # check at low pressure, chi_s = 1
    chi_s = tf.where(pMPA<pp['breaks'][0], pp['coefs'][0,3], chi_s )
       
    # check that there are no negative values at extremely high pressures
    keep_index = tf.cast(chi_s>0, tf.float32)  
    chi_s = chi_s*keep_index
    
    # collect variables into a dictionary
    chi = {'chi_s':chi_s, 'dp':dp, 'cf0':cf0, 'cf1':cf1, 'cf2':cf2}
    
    return chi



def solid_mass_frac_deriv (chi):
    """ calculate derivative for solid mass fraction chi_s, which only depends on pressure"""
    
    # additional 1e-6 to account for pressures in MPa in dp
    dchisdp = 1e-6*(3*chi['cf0']*np.power(chi['dp'],2) + 2*chi['cf1']*chi['dp'] + chi['cf2'])
    
    return dchisdp


def c1c2_deriv (h2o, co2, c, m):
    """
    calculate derivatives for c1, c2 needed for density equation
    !!! chain rule extravaganza !!!
    """
    
    c1 = c['c1']
    c2 = c['c2']
    
    dc1 = {}
    dc2 = {}
    
    # derivatives for c1
    c1_2       = c1*c1
    dc1['dp']  = c1_2*(h2o['dp']  + co2['dp'] )
    dc1['dmh'] = c1_2*(h2o['dmh'] + co2['dmh'])
    
    # derivatives for c2
    rholhd = m['rho_l']/m['rho_hd']
    rholcd = m['rho_l']/m['rho_cd']
        
    c2_2     = c2*c2
        
    dc2dh2od = -c2_2*(rholhd*(h2o['dissolved']*c1_2 + c1) + rholcd* co2['dissolved']*c1_2 )
    dc2dco2d = -c2_2*(rholhd* h2o['dissolved']*c1_2       + rholcd*(co2['dissolved']*c1_2 + c1))
    
    dc2['dp']  = dc2dh2od*h2o['dp']  + dc2dco2d*co2['dp']
    dc2['dmh'] = dc2dh2od*h2o['dmh'] + dc2dco2d*co2['dmh']
   
    return dc1, dc2


def phasefracs_deriv (phi_g, phi, chi, c, m):
    """
    derivatives of phase fractions
    !!! chain rule extravaganza !!!
    """
    
    c12   = c['c1']*c['c2']
    rhosl = m['rho_s']*m['rho_l']
    
    denom  = chi['chi_s']*(m['rho_l']*c12 - m['rho_s']) + m['rho_s']
    denom2 = np.power(denom,2)
    
    dphisdchi =     c12*rhosl*(1-phi_g)/denom2
    dphisdc1  = c['c2']*rhosl*(1-phi_g)*(1-chi['chi_s'])*chi['chi_s']/denom2
    dphisdc2  = c['c1']*rhosl*(1-phi_g)*(1-chi['chi_s'])*chi['chi_s']/denom2
    
    # solid volume fraction derivs
    dphis           = {}
    dphis['dp']     = dphisdchi*chi['dchisdp'] + dphisdc1*c['dc1']['dp'] + dphisdc2*c['dc2']['dp']
    dphis['dphi_g'] = -c12*m['rho_l']*chi['chi_s']/denom
    dphis['dmh']    = dphisdc1*c['dc1']['dmh'] + dphisdc2*c['dc2']['dmh']
    
    # liquid volume fraction derivs
    dphil           = {}
    dphil['dp']     = -c['c2']*dphis['dp']  + (1 - phi['s'] - phi_g)*c['dc2']['dp']
    dphil['dphi_g'] =  c['c2']*(-dphis['dphi_g'] - 1)
    dphil['dmh']    = -c['c2']*dphis['dmh'] + (1 - phi['s'] - phi_g)*c['dc2']['dmh']
    
    return dphis, dphil



def gasdensity_deriv(p, mh, m):
    
    drhog = {}
    drhog['dp']  = mh/m['Rw']/m['T'] + (1-mh)/m['Rc']/m['T']
    drhog['dmh'] = p*(1/m['Rw']/m['T'] - 1/m['Rc']/m['T'])
    
    return drhog



def chambervolatiles (m):
    
    mh_ch = fsolve(chamber_mh, 0.9, args=(m,))
    
    h2o, co2 = solubility(m['p_ch'], mh_ch, m)
    
    h2o['exsolved'] = 1e-2*m['chi_ch']['h2o'] - h2o['dissolved']
    co2['exsolved'] = 1e-6*m['chi_ch']['co2'] - co2['dissolved']
    total_exsolved  = h2o['exsolved'] + co2['exsolved']
    
    c1 = 1/(1 - h2o['dissolved'] - co2['dissolved'])
    c2 = 1/(1 + h2o['dissolved']*c1*(m['rho_l']/m['rho_hd']) + co2['dissolved']*c1*(m['rho_l']/m['rho_cd']))
    c12 = c1*c2
    
    chi = solid_mass_frac(1e-6*np.array(m['p_ch']).reshape(1,1), m['pf'])
    rho_g = m['p_ch']*(mh_ch/(m['Rw']*m['T']) + (1-mh_ch)/(m['Rc']*m['T']))
    
    d = (total_exsolved*m['rho_l']*c12) * (1 - chi['chi_s']*m['rho_l']*c12 / (m['rho_s'] + chi['chi_s']*(c12*m['rho_l'] - m['rho_s'])))
    phi_g_ch = d / (rho_g + d)
    
    return mh_ch, phi_g_ch


def chamber_mh (mh, m):
    # non-linear equation for mh in chamber
    
    gamma    = (1-mh)/(m['B']*mh)
    h2o, co2 = solubility(m['p_ch'], mh, m)
    
    h2o['exsolved'] = 1e-2*m['chi_ch']['h2o'] - h2o['dissolved']
    co2['exsolved'] = 1e-6*m['chi_ch']['co2'] - co2['dissolved']
    
    residual = gamma*h2o['exsolved'] - co2['exsolved']
    
    return residual




def magmaperm (phi_g, degas, m):
    """ Calculate magma permeability using Kozeny-Carman"""
    kmag = m['kc']*np.power(phi_g,3)*degas
    
    return kmag


def wallperm (z, m):
    """calculate wall rock permeability from Manning and Ingebritsen [1999]"""
    
    kwall = m['klw']['top'] / np.power(1e-3*(-z),m['klw']['mi'])
    
    return kwall


def permeability (phi_g, degas, z, m):
    """ Calculate vertical and lateral permeability, 
    lateral is combination of wall rock and magma permeability"""
    
    kmag  = magmaperm(phi_g, degas, m)
    kwall = wallperm(z, m)
    klat  = (m['klw']['L']+m['R']) / (m['R']/kmag + m['klw']['L']/kwall)
    
    k = {}
    k['vert'] = kmag
    k['lat']  = klat

    return k


def gasvels (p, phi_g, dpdz, degas, z, m):
    
    gvel = {}
    
    k = permeability(phi_g, degas, z, m)
    phyd = m['phydro']['slope']*(-z)
    
    gvel['vg'] = {'vel': -k['vert']/m['eta_g']*dpdz}
    gvel['ug'] =  k['lat']/m['eta_g']*(p - phyd)/(m['klw']['L'] + m['R'])
    
    gvel['vg']['dp']     = -k['vert']/m['eta_g']*degas
    gvel['vg']['dphi_g'] = -1/m['eta_g']*dpdz*3*m['kc']*np.power(phi_g,2)*degas
    
    return gvel
    
