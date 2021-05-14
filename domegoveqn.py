import numpy as np
from scipy import special as sp
import constitutive
import utils




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
    F  = np.zeros((Nz,4))
    
    # necessary constitutive relations
    h2o, co2, c, phi, rho = constitutive.density(var['p'], var['phi_g'], var['mh'], m)
    gamma = constitutive.exsolvedco2h2o(var['mh'],m)
    degas = 1.
    gvel  = constitutive.gasvels(var['p'], var['phi_g'], grad['p']['z'], degas, z, m)
    
    
    # mbe--------------------------------------------------------------------------------
    mbe = mbe_vals(z, var['v'], var['phi_g'], grad['p']['z'], 100*h2o['dissolved'], phi, rho['mix'], m)
    
    F[:, m['vi']['v']] = var['v'].reshape(Nz) - mbe['vv'].reshape(Nz) - mbe['vfr'].reshape(Nz)
    
    
    # continuity------------------------------------------------------------------------
        
    # solids and liquids. msl = mass of solids and liquids
    msl           = {}
    msl['dv']     =           m['rho_s']*phi['s']               + m['rho_l']*phi['l']
    msl['dp']     = var['v']*(m['rho_s']*phi['dphis']['dp']     + m['rho_l']*phi['dphil']['dp']    )
    msl['dphi_g'] = var['v']*(m['rho_s']*phi['dphis']['dphi_g'] + m['rho_l']*phi['dphil']['dphi_g'])
    msl['dmh']    = var['v']*(m['rho_s']*phi['dphis']['dmh']    + m['rho_l']*phi['dphil']['dmh']   )
    msl['total']  = msl['dv']*grad['v']['z'] + msl['dp']*grad['p']['z'] + msl['dphi_g']*grad['phi_g']['z'] + msl['dmh']*grad['mh']['z']
    
    F[:,m['vi']['p']] = msl['total'].reshape(Nz)

    # water, separate into liquid and gas parts for slightly more intelligible code
    mh2o = {'l': { }, 'g': { }, 'vg':{}, 'plug':{}}

    # liquid part
    mh2o['l']['dv']     =          m['rho_l']*  h2o['dissolved']*phi['l']              *c['c1']
    mh2o['l']['dp']     = var['v']*m['rho_l']*( h2o['dissolved']*phi['dphil']['dp']    *c['c1'] + h2o['dissolved']*phi['l']*c['dc1']['dp']  + h2o['dp']*phi['l']*c['c1']) 
    mh2o['l']['dphi_g'] = var['v']*m['rho_l']*( h2o['dissolved']*phi['dphil']['dphi_g']*c['c1']  ) 
    mh2o['l']['dmh']    = var['v']*m['rho_l']*( h2o['dissolved']*phi['dphil']['dmh']   *c['c1'] + h2o['dissolved']*phi['l']*c['dc1']['dmh'] + h2o['dmh']*phi['l']*c['c1']) 
    mh2o['l']['total']  = mh2o['l']['dv']*grad['v']['z'] + mh2o['l']['dp']*grad['p']['z'] + mh2o['l']['dphi_g']*grad['phi_g']['z'] + mh2o['l']['dmh']*grad['mh']['z'] 
    
    # gas part
    mh2o['g']['dv']     =           gamma['h']*var['phi_g']*rho['g']
    mh2o['g']['dp']     = var['v']* gamma['h']*var['phi_g']*rho['drhog']['dp']
    mh2o['g']['dphi_g'] = var['v']* gamma['h']*             rho['g']
    mh2o['g']['dmh']    = var['v']*(gamma['h']*var['phi_g']*rho['drhog']['dmh'] + rho['g']*var['phi_g']*gamma['dmh'])
    mh2o['g']['total']  = mh2o['g']['dv']*grad['v']['z'] + mh2o['g']['dp']*grad['p']['z'] + mh2o['g']['dphi_g']*grad['phi_g']['z'] + mh2o['g']['dmh']*grad['mh']['z'] 

    # vertical gas escape part 
    mh2o['vg']['dp']     = gvel['vg']['vel']*mh2o['g']['dp']     + mh2o['g']['dv']*gvel['vg']['dp']
    mh2o['vg']['dphi_g'] = gvel['vg']['vel']*mh2o['g']['dphi_g'] - mh2o['g']['dv']*gvel['vg']['dphi_g']
    mh2o['vg']['dmh']    = gvel['vg']['vel']*mh2o['g']['dmh']
    mh2o['vg']['total']  = mh2o['vg']['dp']*grad['p']['z'] + mh2o['vg']['dphi_g']*grad['phi_g']['z'] + mh2o['vg']['dmh']*grad['mh']['z'] 
    
    # lateral gas escape part
    mh2o['ug'] = 2.*gamma['h']*rho['g']*var['phi_g']*gvel['ug']/m['R']
    
    F[:,m['vi']['phi_g']] = mh2o['l']['total'].reshape(Nz) + mh2o['g']['total'].reshape(Nz) + mh2o['vg']['total'].reshape(Nz) + mh2o['ug'].reshape(Nz)

    
    # carbon dioxide, separate into liquid and gas parts for slightly more intelligible code
    mco2 = {'l': { }, 'g': { }, 'vg':{}, 'plug':{}}
    
    # liquid part
    mco2['l']['dv']     =          m['rho_l']*  co2['dissolved']*phi['l']              *c['c1']
    mco2['l']['dp']     = var['v']*m['rho_l']*( co2['dissolved']*phi['dphil']['dp']    *c['c1'] + co2['dissolved']*phi['l']*c['dc1']['dp']  + co2['dp']*phi['l']*c['c1'] ) 
    mco2['l']['dphi_g'] = var['v']*m['rho_l']*( co2['dissolved']*phi['dphil']['dphi_g']*c['c1'] ) 
    mco2['l']['dmh']    = var['v']*m['rho_l']*( co2['dissolved']*phi['dphil']['dmh']   *c['c1'] + co2['dissolved']*phi['l']*c['dc1']['dmh'] + co2['dmh']*phi['l']*c['c1']) 
    mco2['l']['total']  = mco2['l']['dv']*grad['v']['z'] + mco2['l']['dp']*grad['p']['z'] + mco2['l']['dphi_g']*grad['phi_g']['z'] + mco2['l']['dmh']*grad['mh']['z'] 
    
    # gas part
    mco2['g']['dv']     =           gamma['c']*var['phi_g']*rho['g']
    mco2['g']['dp']     = var['v']* gamma['c']*var['phi_g']*rho['drhog']['dp']
    mco2['g']['dphi_g'] = var['v']* gamma['c']*             rho['g']
    mco2['g']['dmh']    = var['v']*(gamma['c']*var['phi_g']*rho['drhog']['dmh'] - rho['g']*var['phi_g']*gamma['dmh'])
    mco2['g']['total']  = mco2['g']['dv']*grad['v']['z'] + mco2['g']['dp']*grad['p']['z'] + mco2['g']['dphi_g']*grad['phi_g']['z'] + mco2['g']['dmh']*grad['mh']['z'] 
    
    #vertical gas escape part 
    mco2['vg']['dp']    = gvel['vg']['vel']*mco2['g']['dp']     + mco2['g']['dv']*gvel['vg']['dp']
    mco2['vg']['dphi_g']= gvel['vg']['vel']*mco2['g']['dphi_g'] - mco2['g']['dv']*gvel['vg']['dphi_g']
    mco2['vg']['dmh']   = gvel['vg']['vel']*mco2['g']['dmh']
    mco2['vg']['total'] = mco2['vg']['dp']*grad['p']['z'] + mco2['vg']['dphi_g']*grad['phi_g']['z'] + mco2['vg']['dmh']*grad['mh']['z'] 
    
    # lateral gas escape part
    mco2['ug'] = 2*gamma['c']*rho['g']*var['phi_g']*gvel['ug']/m['R']
    
    F[:,m['vi']['mh']] = mco2['l']['total'].reshape(Nz) + mco2['g']['total'].reshape(Nz) + mco2['vg']['total'].reshape(Nz) + mco2['ug'].reshape(Nz)

    
    
    # check if we are in the plug
    iplug = (mbe['vv']/(mbe['vv']+mbe['vfr']) - m['newtonian']['vvfrac_thr']<0).reshape(Nz)
    
    mh2o['plug']['dv']     = gamma['h']  *var['phi_g']
    mh2o['plug']['dp']     = np.zeros((Nz,1))
    mh2o['plug']['dphi_g'] = gamma['h']               *var['v']
    mh2o['plug']['dmh']    = gamma['dmh']*var['phi_g']*var['v']
    mh2o['plug']['total']  = mh2o['plug']['dv']*grad['v']['z'] + mh2o['plug']['dp']*grad['p']['z'] + mh2o['plug']['dphi_g']*grad['phi_g']['z'] + mh2o['plug']['dmh']*grad['mh']['z'] 
    mh2o['plug']['total']  = mh2o['plug']['total'].reshape(Nz)
    
    F[iplug, m['vi']['phi_g']] = mh2o['plug']['total'][iplug]
    
    mco2['plug']['dv']     =  gamma['c']   *var['phi_g']
    mco2['plug']['dp']     =  np.zeros((Nz,1))
    mco2['plug']['dphi_g'] =  gamma['c']               *var['v']
    mco2['plug']['dmh']    = -gamma['dmh']*var['phi_g']*var['v']
    mco2['plug']['total']  = mco2['plug']['dv']*grad['v']['z'] + mco2['plug']['dp']*grad['p']['z'] + mco2['plug']['dphi_g']*grad['phi_g']['z'] + mco2['plug']['dmh']*grad['mh']['z'] 
    mco2['plug']['total']  = mco2['plug']['total'].reshape(Nz)
    
    F[iplug, m['vi']['mh']] = mco2['plug']['total'][iplug]
    
    # assign boundary conditions
    F = assign_bcs(F, var, m)
    
    return F, msl, mh2o, mco2, iplug, mbe, phi, h2o, co2, rho





def mbe_vals (z, v, phi_g, dpdz, h2o_d, phi, rho, m):
    
    phi_s_eta = phi['s']/(1 - phi_g)
    sig_c = m['sig']['slope']*(-z) + m['sig']['offset']
    gdot  = 2*np.absolute(v)/m['R']
    
    eta   = constitutive.viscosity(h2o_d, phi_s_eta, gdot, m)
    tau_R = -0.5*m['R']*(dpdz + rho*m['g'])
    
    vfr = m['fr']['A']*np.sinh(tau_R/m['fr']['a']/sig_c)
    vv  = tau_R*m['R']/4/eta['mix']
    
    mbe = {'vfr': vfr, 'eta':eta['mix'], 'vv':vv, 'tau_R':tau_R}
    return mbe
    

def assign_bcs (F, var, m):
    
    mh_ch, phi_g_ch = constitutive.chambervolatiles(m)
    
    F[ 0,m['vi']['p']]     = var['p'][ 0]    - m['p_ch']
    F[-1,m['vi']['v']]     = var['p'][-1]    - m['p_top']
    F[ 0,m['vi']['phi_g']] = var['phi_g'][0] - phi_g_ch
    F[ 0,m['vi']['mh']]    = var['mh'][0]    - mh_ch
    
    
    return F
    
    
