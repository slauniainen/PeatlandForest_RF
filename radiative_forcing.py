import numpy as np
import pandas as pd

def radiative_forcing(fghg, fdir=None, Scen='T4-SSP1-2.6', t=None, f=1e6):
    """
    Computes radiative forcing of GHG's and albedo changes using REFUGE 4 methodology
    Args:
        ghg_flux - CO2, CH4 and N2O net emissions/sinks (Mton gas a-1). index = year
        fdir - direct surface heating (W m-2 land)
        Scen - reference concentration pathway (T4-SSP1-2.6, T8-SSP4-3.4, T5-SSP2-4.5 
        f - multiplier for numerics
    Returns:
        RF
    """

    fghg = fghg * f

    #conversion factors from stocks to concentrations
    cf_co2 = 0.00012864493996569469 # ppm MtCO2-1
    cf_ch4 = 0.36395399621487845 # ppb MtCH4-1
    cf_n2o = 0.13287970 # ppb MtN2O-1

    if not t:
        t = fghg.index.values
    
    # compute delta stocks from fluxes
    Sc = delta_stock('CO2', fghg['CO2'].values, t)
    Sm = delta_stock('CH4', fghg['CH4'].values, t)
    Sn = delta_stock('N2O', fghg['N2O'].values, t)

    # compute concentration changes from delta stocks
    deltaCc = Sc * cf_co2
    deltaCn = Sn * cf_n2o
    # CH4, adjustments due to atm reactions with N2O
    a = -0.36 # mol CH4 mol N2O-1
    b = 2.74 # molar mass ratio
    deltaCm = (Sm + a/b * Sn) * cf_ch4

    # reference concentrations
    if Scen == 'T4-SSP1-2.6':
        Cref = pd.read_excel(r'Reference_concentrations.xlsx', skiprows=3, usecols=[0, 1, 2, 3], index_col=0, header=None, names=['Year', 'CO2', 'CH4', 'N2O'])
    if Scen == 'T8-SSP4-3.4':
        Cref = pd.read_excel(r'Reference_concentrations.xlsx', skiprows=3, usecols=[0, 4, 5, 6], index_col=0, header=None, names=['Year', 'CO2', 'CH4', 'N2O'])
    if Scen == 'T5-SSP2-4.5':
        Cref = pd.read_excel(r'Reference_concentrations.xlsx', skiprows=3, usecols=[0, 7, 8, 9], index_col=0, header=None, names=['Year', 'CO2', 'CH4', 'N2O'])
    
    Cref = Cref.loc[fghg.index]

    Cc = Cref['CO2'].values
    Cm = Cref['CH4'].values
    Cn = Cref['N2O'].values

    #testing!
    #Cc[:] = Cc[0]
    #Cm[:] = Cm[0]
    #Cn[:] = Cn[0]

    # radiative forcings from concentrations
    # baseline
    RFt0, RFc0, RFm0, RFn0 = GHG_RF(Cc, Cm, Cn) 
    # baseline + land-use
    RFt, RFc, RFm, RFn = GHG_RF(Cc + deltaCc, Cm + deltaCm, Cn + deltaCn)
    
    rf0 = np.array([RFt0, RFc0, RFm0, RFn0]).transpose()
    rf = np.array([RFt, RFc, RFm, RFn]).transpose()
    # difference == impact of LU
    deltaRF = rf - rf0
    RF = pd.DataFrame(data=deltaRF, columns=[['Tot', 'CO2', 'CH4', 'N2O']], index=fghg.index)
    
    return RF/f #, Sc/f, Sm/f, Sn/f

def GHG_RF(Cc, Cm, Cn):
    """"
    Radiative forcing [W m2 (earth)] from concentrations
    Args:
        Cc - CO2 [ppm]
        Cm - CH4 [ppb]
        Cn - N2O [ppb]
    Returns:
        RF - total radiative forcing from all ghg's
        RFc - from CO2
        RFm - from CH4
        RFn - from N2O
    """

    #CO2
    a1 = -2.4785e-7; b1 = 0.00075906; c1 = -0.0021492; d1 = 5.2488; C0 = 277.15; fca = 1.05
    
    alpha = d1 + a1*(Cc - C0)**2 + b1 * (Cc - C0)
    alpha_n = c1*np.sqrt(Cn)

    RFc = (alpha + alpha_n)* np.log(Cc/C0)*fca

    #N2O
    a2 = -0.00034197; b2 = 0.00025455; c2 = -0.00024357; d2 = 0.12173; N0 = 273.87; fna = 1.07

    RFn = (a2*np.sqrt(Cc) + b2*np.sqrt(Cn) + c2*np.sqrt(Cm) + d2) * (np.sqrt(Cn) - np.sqrt(N0))*fna

    #CH4
    a3 = -8.9603e-5; b3 = -0.00012462; d3 = 0.045194; M0 = 731.41; fma = 0.86

    RFm = (a3*np.sqrt(Cm) + b3*np.sqrt(Cn) + d3) * (np.sqrt(Cm) - np.sqrt(M0)) * fma

    RFtot = RFc + RFn + RFm

    return RFtot, RFc, RFm, RFn


def delta_stock(gas, F, t):
    """"
    Computes change in atm. stock of substance s due to annual pulse emissions (F, Mton gas a-1)
    Args:
        gas - 'CO2', 'CH4', 'N2O' (str)
        F - annual net flux to atm. (Mton gas a-1, array)
        tspan - timespan
    Returns
        deltaS- change in atmospheric stock due to emissions (Mton gas)
    """

    # #conversion factors from stocks to concentrations
    # cf = {'CO2': 0.00012864493996569469, # ppm MtCO2-1
    #       'CH4': 0.36395399621487845, # ppb MtCH4-1
    #       'N2O': 0.13287970 # ppb MtN2O-1
    #      }
    
    nsteps = len(t)
    if len(F) < nsteps:
        F = np.pad(F, (0, nsteps - len(F)), 'constant', constant_values=0.0)

    # compute change in stocks of CO2, CH4 and N20
    S = np.zeros(nsteps)

    if gas == 'CH4':
        tau_ch4 = 11.8

        S0 = 0.0
        for j in range(nsteps):
            S[j] = F[j] * np.exp(-0.5 / tau_ch4) + S0 * np.exp(-1.0/ tau_ch4) # current + past emissions
            S0 = S[j]
    if gas == 'N2O':
        tau_n2o = 109.0
        S0 = 0.0
        for j in range(nsteps):
            S[j] = F[j] * np.exp(-0.5 / tau_n2o) + S0 * np.exp(-1.0/ tau_n2o) # current + past emissions
            S0 = S[j]

    if gas == 'CO2': # emissions are divided into three pathways that decay at different rates
        a =[0.2173, 0.224, 0.2824, 0.2763]
        tau_co2 = [394.4, 36.54, 4.304]

        c0 = 0.0; c1 = 0.0; c2 = 0.0; c3 = 0.0
        for j in range(nsteps):
            c0 = c0 + a[0] * F[j]
            c1 = F[j]*a[1] * np.exp(-0.5/tau_co2[0]) + c1 * np.exp(-1.0/tau_co2[0])
            c2 = F[j]*a[2] * np.exp(-0.5/tau_co2[1]) + c2 * np.exp(-1.0/tau_co2[1])
            c3 = F[j]*a[3] * np.exp(-0.5/tau_co2[2]) + c3 * np.exp(-1.0/tau_co2[2])
            S[j] = c0 + c1 + c2 + c3

    return S


def irf(gas, t):
    """
    Time-decay of CO2, CH4 or N2O pulse emission with in the atmosphere
    REFUGE4-documentation eq. (1-2)
    Args:
        gas - 'CO2', 'CH4', 'N2O'
        t - time span (a, array)
    Returns:
        co2_stock
    """
    # conversion from emission to well-mixed atm. concentration
    #cf = 2.12# GtonCO2 / ppm 
    # parameters
    a =[0.2173, 0.224, 0.2824, 0.2763]
    tau_co2 = [394.4, 36.54, 4.304]

    tau_ch4 = 11.8
    tau_n2o = 109.0

    if gas == 'CO2':
        y = a[0] + a[1]*np.exp(-t / tau_co2[0]) + a[2] * np.exp(-t / tau_co2[1]) + a[3] * np.exp(-t / tau_co2[2])
    elif gas == 'CH4':
        y = np.exp(-t/tau_ch4)
    elif gas == 'N2O':
        y = np.exp(-t / tau_n2o)
    else:
        print('irf: unknown gas')
    return y


def albedo_delta_rf(vol):
    """
    Lohila et al. Fig 2 & 7; 
    """
    a = -2.1e-14*0.65
 
    b = 0.53
    c = 0.04

    # range 5m3/ha -->
    f = a*(1 - b*np.exp(-c*vol))
    # linear below 5m3/ha
    #ix = vol <= 5.0
    #f[ix] = vol[ix] * a/5*(1 - b*np.exp(-c*5.0))

    # linear below 1m3/ha
    ix = vol < 3.0
    f[ix] = vol[ix] * a/3.0*(1 - b*np.exp(-c*3.0))

    f = f.reshape(len(f),1)
    
    return f
