import numpy as np
from scipy import special as sp
from constitutive import *




def ssc (z, var, grad, m):
    """
    order of variables: p, v, phi_g, mh
    order of equations (to be consistent with variable order and have diagonals populated)
        1) solids and liquids continuity
        2) mbe
        3) h2o continuity
        4) co2 continuity
    """
    
    Nz = len(z)
    J  = np.zeros((Nz,4,8))
    Q  = np.zeros((Nz,4))
    
    # necessary constitutive relations
    h2o, co2, c, phi, rho = density(var['p'], var['phi_g'], var['mh'], m)
    gamma = exsolvedco2h2o(var['mh'],m)
    degas = 1.
    gvel = gasvels(var['p'], var['phi_g'], grad['p']['z'], degas, z, m)
    
    
    
    
    # mbe--------------------------------------------------------------------------------
    mbe = mbe_vals(z, var['v'], var['phi_g'], grad['p']['z'], 100*h2o['dissolved'], phi, rho['mix'], m)
    
    Q[:, m['vi']['v']] = var['v'].reshape(Nz) - mbe['vv'].reshape(Nz) - mbe['vfr'].reshape(Nz)
    
    
    # continuity------------------------------------------------------------------------
    
    # first reshape velocities 
    vrshp  = var['v'].reshape(Nz)
    vgrshp = gvel['vg']['vel'].reshape(Nz)
    gvel['vg'] = mrshp(gvel['vg'], Nz)
    
    
    # solids and liquids. msl = mass of solids and liquids
    msl           = {}
    msl['dv']     = m['rho_s']*phi['s']               + m['rho_l']*phi['l']
    msl['dp']     = m['rho_s']*phi['dphis']['dp']     + m['rho_l']*phi['dphil']['dp']
    msl['dphi_g'] = m['rho_s']*phi['dphis']['dphi_g'] + m['rho_l']*phi['dphil']['dphi_g']
    msl['dmh']    = m['rho_s']*phi['dphis']['dmh']    + m['rho_l']*phi['dphil']['dmh']
    
    # reshape 
    msl = mrshp(msl, Nz)

    J[:, m['vi']['p'], m['vi']['vz']]     =       msl['dv']
    J[:, m['vi']['p'], m['vi']['pz']]     = vrshp*msl['dp']
    J[:, m['vi']['p'], m['vi']['phi_gz']] = vrshp*msl['dphi_g']
    J[:, m['vi']['p'], m['vi']['mhz']]    = vrshp*msl['dmh']
    
    
    
    
    # water, separate into liquid and gas parts for slightly more intelligible code
    mh2o = {'l': { }, 'g': { }, 'plug':{}}

    # liquid part
    mh2o['l']['dv']     = m['rho_l']*  h2o['dissolved']*phi['l']              *c['c1']
    mh2o['l']['dp']     = m['rho_l']*( h2o['dissolved']*phi['dphil']['dp']    *c['c1'] + h2o['dissolved']*phi['l']*c['dc1']['dp']  + h2o['dp']*phi['l']*c['c1']) 
    mh2o['l']['dphi_g'] = m['rho_l']*( h2o['dissolved']*phi['dphil']['dphi_g']*c['c1'] ) 
    mh2o['l']['dmh']    = m['rho_l']*( h2o['dissolved']*phi['dphil']['dmh']   *c['c1'] + h2o['dissolved']*phi['l']*c['dc1']['dmh'] + h2o['dmh']*phi['l']*c['c1']) 
    
    # gas part
    mh2o['g']['dv']     = gamma['h']*var['phi_g']*rho['g']
    mh2o['g']['dp']     = gamma['h']*var['phi_g']*rho['drhog']['dp']
    mh2o['g']['dphi_g'] = gamma['h']*             rho['g']
    mh2o['g']['dmh']    = gamma['h']*var['phi_g']*rho['drhog']['dmh'] + rho['g']*var['phi_g']*gamma['dmh']
    
    # reshape 
    mh2o['l'] = mrshp(mh2o['l'], Nz)
    mh2o['g'] = mrshp(mh2o['g'], Nz)
    
    J[:, m['vi']['phi_g'], m['vi']['vz']]     =        mh2o['l']['dv']    + mh2o['g']['dv']
    J[:, m['vi']['phi_g'], m['vi']['pz']]     = vrshp*(mh2o['l']['dp']    + mh2o['g']['dp']    ) #+ vgrshp*mh2o['g']['dp']     + mh2o['g']['dv']*gvel['vg']['dp']
    J[:, m['vi']['phi_g'], m['vi']['phi_gz']] = vrshp*(mh2o['l']['dphi_g']+ mh2o['g']['dphi_g']) #+ vgrshp*mh2o['g']['dphi_g'] - mh2o['g']['dv']*gvel['vg']['dphi_g']
    J[:, m['vi']['phi_g'], m['vi']['mhz']]    = vrshp*(mh2o['l']['dmh']   + mh2o['g']['dmh']   ) #+ vgrshp*mh2o['g']['dmh']
        
    Qh = 2.*gamma['h']*rho['g']*var['phi_g']*gvel['ug']/m['R']
    Q[:,m['vi']['phi_g']] = Qh.reshape(Nz)
    
    
    
    # carbon dioxide, separate into liquid and gas parts for slightly more intelligible code
    mco2 = {'l': { }, 'g': { }, 'plug':{}}
    
    
    # liquid part
    mco2['l']['dv']     = m['rho_l']*  co2['dissolved']*phi['l']              *c['c1']
    mco2['l']['dp']     = m['rho_l']*( co2['dissolved']*phi['dphil']['dp']    *c['c1'] + co2['dissolved']*phi['l']*c['dc1']['dp']  + co2['dp']*phi['l']*c['c1'] ) 
    mco2['l']['dphi_g'] = m['rho_l']*( co2['dissolved']*phi['dphil']['dphi_g']*c['c1'] ) 
    mco2['l']['dmh']    = m['rho_l']*( co2['dissolved']*phi['dphil']['dmh']   *c['c1'] + co2['dissolved']*phi['l']*c['dc1']['dmh'] + co2['dmh']*phi['l']*c['c1']) 
    
    # gas part
    mco2['g']['dv']     = gamma['c']*var['phi_g']*rho['g']
    mco2['g']['dp']     = gamma['c']*var['phi_g']*rho['drhog']['dp']
    mco2['g']['dphi_g'] = gamma['c']*             rho['g']
    mco2['g']['dmh']    = gamma['c']*var['phi_g']*rho['drhog']['dmh'] - rho['g']*var['phi_g']*gamma['dmh']
    
    # reshape 
    mco2['l'] = mrshp(mco2['l'], Nz)
    mco2['g'] = mrshp(mco2['g'], Nz)
    
    J[:, m['vi']['mh'], m['vi']['vz']]     =        mco2['l']['dv']    + mco2['g']['dv']
    J[:, m['vi']['mh'], m['vi']['pz']]     = vrshp*(mco2['l']['dp']    + mco2['g']['dp'] )    #+ vgrshp*mco2['g']['dp']     + mco2['g']['dv']*gvel['vg']['dp']
    J[:, m['vi']['mh'], m['vi']['phi_gz']] = vrshp*(mco2['l']['dphi_g']+ mco2['g']['dphi_g']) #+ vgrshp*mco2['g']['dphi_g'] - mco2['g']['dv']*gvel['vg']['dphi_g']
    J[:, m['vi']['mh'], m['vi']['mhz']]    = vrshp*(mco2['l']['dmh']   + mco2['g']['dmh'])    #+ vgrshp*mco2['g']['dmh']
    
    Qc = 2*gamma['c']*rho['g']*var['phi_g']*gvel['ug']/m['R']
    Q[:,m['vi']['mh']] = Qc.reshape(Nz)
    
    
    
    # check if we are in the plug
    iplug = (mbe['vv']/(mbe['vv']+mbe['vfr']) - m['newtonian']['vvfrac_thr']<0).reshape(Nz)
    
    mh2o['plug']['dv']     = gamma['h']*var['phi_g']
    mh2o['plug']['dp']     = np.zeros((Nz,1))
    mh2o['plug']['dphi_g'] = gamma['h']*var['v']
    mh2o['plug']['dmh']    = gamma['dmh']*var['phi_g']*var['v']
    mh2o['plug']           = mrshp(mh2o['plug'], Nz)
    
    J[iplug,m['vi']['phi_g'],m['vi']['vz']]     = mh2o['plug']['dv'][iplug]
    J[iplug,m['vi']['phi_g'],m['vi']['pz']]     = mh2o['plug']['dp'][iplug]
    J[iplug,m['vi']['phi_g'],m['vi']['phi_gz']] = mh2o['plug']['dphi_g'][iplug]
    J[iplug,m['vi']['phi_g'],m['vi']['mhz']]    = mh2o['plug']['dmh'][iplug]
    
    mco2['plug']['dv']     = gamma['c']*var['phi_g']
    mco2['plug']['dp']     = np.zeros((Nz,1))
    mco2['plug']['dphi_g'] = gamma['c']*var['v']
    mco2['plug']['dmh']    = -gamma['dmh']*var['phi_g']*var['v']
    mco2['plug']           = mrshp(mco2['plug'], Nz)
    
    J[iplug,m['vi']['mh'],m['vi']['vz']]     = mco2['plug']['dv'][iplug]
    J[iplug,m['vi']['mh'],m['vi']['pz']]     = mco2['plug']['dp'][iplug]
    J[iplug,m['vi']['mh'],m['vi']['phi_gz']] = mco2['plug']['dphi_g'][iplug]
    J[iplug,m['vi']['mh'],m['vi']['mhz']]    = mco2['plug']['dmh'][iplug]
    
    
    return J, Q, iplug, mbe, phi, h2o, co2, rho





def mbe_vals (z, v, phi_g, dpdz, h2o_d, phi, rho, m):
    
    phi_s_eta = phi['s']/(1 - phi_g)
    sig_c = m['sig']['slope']*(-z) + m['sig']['offset']
    gdot  = 2*np.absolute(v)/m['R']
    
    eta   = viscosity(h2o_d, phi_s_eta, gdot, m)
    tau_R = -0.5*m['R']*(dpdz + rho*m['g'])
    
    vfr = m['fr']['A']*np.sinh(tau_R/m['fr']['a']/sig_c)
    vv  = tau_R*m['R']/4/eta['mix']
    
    mbe = {'vfr': vfr, 'eta':eta['mix'], 'vv':vv, 'tau_R':tau_R}
    return mbe
    
    
    
    
    
def mrshp (m, Nz):
    m['dp']     = m['dp'].reshape(Nz)
    m['dphi_g'] = m['dphi_g'].reshape(Nz)

    if 'dv' in m:
        m['dv']    = m['dv'].reshape(Nz)
        
    if 'dmh' in m:
        m['dmh']    = m['dmh'].reshape(Nz)
    
    return m

