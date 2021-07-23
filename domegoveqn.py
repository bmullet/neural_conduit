import numpy as np
from scipy import special as sp
import constitutive
import tensorflow as tf
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
    J  = tf.zeros((Nz,4,8)) #TODO delete
    F  = tf.zeros((Nz,4))
    
    # Some helper functions
    # Get a new (all zeros) mask of F
    F_mask = lambda: np.zeros_like(F)
    # Convert (probably a mask) to tensorflow
    np_to_tf = lambda tensor: tf.constant(tensor, dtype=tf.float32)
    
    
    # necessary constitutive relations
    h2o, co2, c, phi, rho = constitutive.density(var['p'], var['phi_g'], var['mh'], m)
    gamma = constitutive.exsolvedco2h2o(var['mh'],m)
    degas = 1.0
    gvel, k  = constitutive.gasvels(var['p'], var['phi_g'], grad['p']['z'], degas, z, m)
    
    
    # mbe--------------------------------------------------------------------------------
    mbe = mbe_vals(z, var['v'], var['phi_g'], grad['p']['z'], 100*h2o['dissolved'], phi, rho['mix'], m)
    
    # What we want to do, but tensorflow is dumb
    #F[:, m['vi']['v']] = tf.reshape(var['v'] - mbe['vv'] - mbe['vfr'], [Nz])
    
    # How we can do it instead
    # Get new mask
    mbemask = F_mask()
    mbemask[:, m['vi']['v']] = 1.0 #index assingment OK on numpy array
    mbemask = np_to_tf(mbemask)
    
    mbeupdate = tf.reshape(var['v'] - mbe['vv'] - mbe['vfr'], [Nz,1]) 
    
    # check for bad values
    #mbeupdate = tf.where(tf.math.is_nan(mbeupdate), 1e8, mbeupdate)
    #mbeupdate = tf.where(tf.math.is_inf(mbeupdate), 1e8, mbeupdate)
    
    F = F + mbemask*mbeupdate # update is 1-D while mask is 2-D, so update will broadcast
    
    # continuity------------------------------------------------------------------------
    # continuity of solids and liquids
    msl = slcont(var, grad, phi, m)
    
    mslmask = F_mask()
    mslmask[:,m['vi']['p']] = 1.0
    mslmask = np_to_tf(mslmask)
    mslupdate = tf.reshape(msl['total'], [Nz,1])
    F = F + mslmask*mslupdate
    
    
    
    # continuity for water
    mh2o =  h2ocont(var, grad, rho, phi, h2o, c, gamma, gvel, k, m)
    
    mh2omask = F_mask()
    mh2omask[:,m['vi']['phi_g']] = 1.0
    mh2omask = np_to_tf(mh2omask)
    
    mh2oviscupdate = tf.reshape(mh2o['l']['total'] + mh2o['g']['total'] + mh2o['vg']['total'] + mh2o['ug'], [Nz,1])
    mh2oplugupdate = tf.reshape(mh2o['plug']['total'], [Nz,1])
    
    # check if we are in the plug
    mh2oupdate = tf.where((mbe['vv']/(mbe['vv']+mbe['vfr']) < m['vvfrac_thr']), mh2oplugupdate, mh2oviscupdate)
    
    F = F + mh2omask*mh2oupdate
    
    
    # continuity for co2
    mco2 = co2cont (var, grad, rho, phi, h2o, c, gamma, gvel, k, m)
    
    mco2mask = F_mask()
    mco2mask[:,m['vi']['mh']] = 1.0
    mco2mask = np_to_tf(mco2mask)
    
    mco2viscupdate = tf.reshape(mco2['l']['total'] + mco2['g']['total'] + mco2['vg']['total'] +  mco2['ug'], [Nz,1])
    mco2plugupdate = tf.reshape(mco2['plug']['total'], [Nz,1])
    
    mco2update = tf.where((mbe['vv']/(mbe['vv']+mbe['vfr']) < m['vvfrac_thr']), mco2plugupdate, mco2viscupdate)
    
    F = F + mco2mask*mco2update
        
    return F



    
def assign_bcs (F, var, m):
    # this is a legacy function that I used for assigning BCs, but we don't need this in physics loss
    
    mh_ch, phi_g_ch = constitutive.chambervolatiles(m)
    
    F[ 0,m['vi']['p']]     = var['p'][ 0]    - m['p_ch']
    F[-1,m['vi']['v']]     = var['p'][-1]    - m['p_top']
    F[ 0,m['vi']['phi_g']] = var['phi_g'][0] - phi_g_ch
    F[ 0,m['vi']['mh']]    = var['mh'][0]    - mh_ch
    
    
    return F
    
    
    

def mbe_vals (z, v, phi_g, dpdz, h2o_d, phi, rho, m):
    
    # TODO: need to address what happens if phi_s is nan, i.e. in cases where phi_g <0!
    phi_s_eta = phi['s']/(1 - phi_g)
    sig_c = m['sig']['slope']*(-z) + m['sig']['offset']
    gdot  = 2*np.absolute(v)/m['R']
    
    eta   = constitutive.viscosity(h2o_d, phi_s_eta, gdot, m)
    tau_R = -0.5*m['R']*(dpdz + rho*m['g'])
    
    vv  = tau_R*m['R']/4/eta['mix']
    vfr = m['fr']['A']*np.sinh(tau_R/m['fr']['a']/sig_c)
    
    # cap vfr at 1 so that sinh function doesn't blow up ridiculously. Beyond this, vfr increases linearly with tauR
    # interpretation: if condition is true, take from x; if false take from y
    #alpha = 1/m['fr']['A'] - np.arcsinh(1/m['fr']['A']);
    #vfrlin = m['fr']['A']*(alpha + tau_R/m['fr']['a']/sig_c)
    #vfr = tf.where(vfr1<1, vfr1, vfrlin)
    
    mbe = {'vfr': vfr, 'eta':eta['mix'], 'vv':vv, 'tau_R':tau_R}
    
    return mbe






def slcont (var, grad, phi, m):
    
    # continuity of solids and liquids. msl = mass of solids and liquids
    msl           = {}
    msl['dv']     =           m['rho_s']*phi['s']               + m['rho_l']*phi['l']
    msl['dp']     = var['v']*(m['rho_s']*phi['dphis']['dp']     + m['rho_l']*phi['dphil']['dp']    )
    msl['dphi_g'] = var['v']*(m['rho_s']*phi['dphis']['dphi_g'] + m['rho_l']*phi['dphil']['dphi_g'])
    msl['dmh']    = var['v']*(m['rho_s']*phi['dphis']['dmh']    + m['rho_l']*phi['dphil']['dmh']   )
    msl['total']  = msl['dv']*grad['v']['z'] + msl['dp']*grad['p']['z'] + msl['dphi_g']*grad['phi_g']['z'] + msl['dmh']*grad['mh']['z']
    
    return msl







def h2ocont (var, grad, rho, phi, h2o, c, gamma, gvel, k, m):
    
    # continuity for water, separate into liquid and gas parts, also plug 
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
    mh2o['g']['dmh']    = var['v']*(gamma['h']*var['phi_g']*rho['drhog']['dmh'] + gamma['dmh']*var['phi_g']*rho['g'])
    mh2o['g']['total']  = mh2o['g']['dv']*grad['v']['z'] + mh2o['g']['dp']*grad['p']['z'] + mh2o['g']['dphi_g']*grad['phi_g']['z'] + mh2o['g']['dmh']*grad['mh']['z'] 

    # vertical gas escape part 
    mh2o['vg']['dvg']    =-gamma['h']*var['phi_g']*rho['g']/m['eta_g']*( k['vert']*grad['p']['z2'] + grad['p']['z']*3*m['kc']*var['phi_g']*var['phi_g']*grad['phi_g']['z'] )
    mh2o['vg']['dp']     = gvel['vg']['vel']*gamma['h']*var['phi_g']*rho['drhog']['dp']
    mh2o['vg']['dphi_g'] = gvel['vg']['vel']*gamma['h']*             rho['g']
    mh2o['vg']['dmh']    = gvel['vg']['vel']*gamma['h']*var['phi_g']*rho['drhog']['dmh'] + gvel['vg']['vel']*gamma['dmh']*var['phi_g']*rho['g']
    mh2o['vg']['total']  = mh2o['vg']['dp']*grad['p']['z'] + mh2o['vg']['dphi_g']*grad['phi_g']['z'] + mh2o['vg']['dmh']*grad['mh']['z'] + mh2o['vg']['dvg'] 
    
    # lateral gas escape part
    mh2o['ug'] = 2.*gamma['h']*rho['g']*var['phi_g']*gvel['ug']/m['R']
    
    # plug part
    mh2o['plug']['dv']     = gamma['h']  *var['phi_g']
    mh2o['plug']['dp']     = np.zeros_like(var['p'])
    mh2o['plug']['dphi_g'] = gamma['h']               *var['v']
    mh2o['plug']['dmh']    = gamma['dmh']*var['phi_g']*var['v']
    mh2o['plug']['total']  = mh2o['plug']['dv']*grad['v']['z'] + mh2o['plug']['dp']*grad['p']['z'] + mh2o['plug']['dphi_g']*grad['phi_g']['z'] + mh2o['plug']['dmh']*grad['mh']['z'] 
    
    return mh2o








def co2cont (var, grad, rho, phi, co2, c, gamma, gvel, k, m):
    
    # continuity for carbon dioxide, separate into liquid and gas parts, also plug 
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
    mco2['g']['dmh']    = var['v']*(gamma['c']*var['phi_g']*rho['drhog']['dmh'] + rho['g']*var['phi_g']*(-gamma['dmh']))
    mco2['g']['total']  = mco2['g']['dv']*grad['v']['z'] + mco2['g']['dp']*grad['p']['z'] + mco2['g']['dphi_g']*grad['phi_g']['z'] + mco2['g']['dmh']*grad['mh']['z'] 
    
    # vertical gas escape part 
    mco2['vg']['dvg']    =-gamma['c']*var['phi_g']*rho['g']/m['eta_g']*( k['vert']*grad['p']['z2'] + grad['p']['z']*3*m['kc']*var['phi_g']*var['phi_g']*grad['phi_g']['z'] )
    mco2['vg']['dp']     = gvel['vg']['vel']*gamma['c']*var['phi_g']*rho['drhog']['dp']
    mco2['vg']['dphi_g'] = gvel['vg']['vel']*gamma['c']*             rho['g']
    mco2['vg']['dmh']    = gvel['vg']['vel']*gamma['c']*var['phi_g']*rho['drhog']['dmh'] + gvel['vg']['vel']*(-gamma['dmh'])*var['phi_g']*rho['g']
    mco2['vg']['total']  = mco2['vg']['dp']*grad['p']['z'] + mco2['vg']['dphi_g']*grad['phi_g']['z'] + mco2['vg']['dmh']*grad['mh']['z'] + mco2['vg']['dvg'] 
    
    # lateral gas escape part
    mco2['ug'] = 2*gamma['c']*rho['g']*var['phi_g']*gvel['ug']/m['R']
    
    # plug part
    mco2['plug']['dv']     =  gamma['c']   *var['phi_g']
    mco2['plug']['dp']     =  np.zeros_like(var['p'])
    mco2['plug']['dphi_g'] =  gamma['c']               *var['v']
    mco2['plug']['dmh']    = -gamma['dmh']*var['phi_g']*var['v']
    mco2['plug']['total']  = mco2['plug']['dv']*grad['v']['z'] + mco2['plug']['dp']*grad['p']['z'] + mco2['plug']['dphi_g']*grad['phi_g']['z'] + mco2['plug']['dmh']*grad['mh']['z'] 
    
    return mco2







