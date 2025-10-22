#from src.stand_carbon import wood_pools, soilCO2
import numpy as np
import pandas as pd
from radiative_forcing import radiative_forcing

# --- Wood and residue pool model

class wood_pools():
    """"
    Wood products or residcue C pool dynamics. Single pool with constant 1st-order decay
    """
    def __init__(self, S0, tau):
        """
        Args:
            S0 - intial storage
            tau - time constant (yr)
        Returns:
            self
        """
        self.S = S0
        self.k = 1/tau

    def update(self, Fin, dt=1):
        """
        updates C pool
        Args: 
            Fin - C flux to pool
            dt - timestep (yr)
        Returns:
            self.S
            Fout - C flux from pool
        """
        Fout = self.S*(1-np.exp(-self.k*dt))
        self.S += Fin - Fout

        return self.S, Fout

# --- Drained forest peatland soil CO2 balance as function of WT

def soilCO2(x, ftype, wtd_para={'a': -50.6, 'b': 29.36, 'c': 0.98698}):
    """
    Soil CO2 balance (g CO2 m-2 a-1) from stem volume (x)
    Based on combining WTD-Vol response from Sarkkola et al. 2010 Fig.4
    and Ojanen & Minkkinen, 2019 eq. 3-4.
    Args: 
        x - stem volume (m3 ha-1)
        ftype - 'FNR' - nutrient rich or 'FNP' nutrient poor
    Returns:
        f - soil C balance (g C m-2)
    """
    # WTD as function of Vol
    #a = -50.56; b = 29.356; c = 0.98698
    a = wtd_para['a']
    b = wtd_para['b']
    c = wtd_para['c']

    wtd = a + b * np.power(c, x)
    
    if ftype == 'FNR':
        f = -115.0  -12.0 * wtd
    elif ftype == 'FNP':
        f = -259.0 - 6.0 * wtd
    
    f = f * (12/44)
    #print(np.max(x), np.min(wtd), np.max(f))
    return f


# --- compute C stocks and net CO2, CH4 and N2O fluxes to atmosphere

def compute_stocks_fluxes(dat, ftype=None, wtd_para={'a': -50.6, 'b': 29.36, 'c': 0.98698}, fyear=2025):

    # soil net GHG balances (Laine et al. 2024 Table 1). Unit = g gas m-2 a-1
    Fsoil = {'FNR': {'CO2': 265.0, 'CH4': 0.34, 'N2O': 0.23}, # forest, nutrient rich
            'FNP': {'CO2': -45.0, 'CH4': 0.34, 'N2O': 0.08}, # forest, nutrient poor
            'RSM': {'CO2': -91.0, 'CH4': 1.70, 'N2O': 0.10}, # restored, spruce mire
            'RPM': {'CO2': -97.0, 'CH4': 4.80, 'N2O': 0.03}, # restored, pine mire
            'RME': {'CO2': -104.0, 'CH4': 15.00, 'N2O': 0.10}, # restored, open eu/mesotrophic
            'ROL': {'CO2': -124.0, 'CH4': 22.00, 'N2O': 0.03}, # restored, open oligotrophic
            'ROM': {'CO2': -95.0, 'CH4': 9.70, 'N2O': 0.03}, # restored, open ombotrophics
           }
    
    # mean lifetimes
    # wood product pools
    tau_wpl = 30.0
    tau_wps = 3.0

    # saw yield
    alpha_saw = 0.4

    # residue pools
    tau_fol = 3.3
    tau_fwd = 7.0
    tau_cwd = 30.0
    tau_cwdr = 300.0

    cols = ['year', 'Age', 'Vol', 'BA', 'C_tree', 'C_resid', 'C_WP_short', 'C_WP_long', 'C_soil', 
            'F_tree', 'F_resid', 'F_soil', 'F_WP_short', 'F_WP_long', 'F_CH4', 'F_N2O']

    N = len(dat)
    #time = np.arange(fyear, fyear + N, 1)

    res = pd.DataFrame(data=np.zeros((N, len(cols))), columns=cols, index=dat.index)

    # forestry scenarios
    if ftype in ['FNR', 'FNP']:
        res[['year', 'Age', 'Vol', 'BA']]  = dat[['year', 'DomAge', 'vol', 'BA']]
        
        # all in g C m-2 or g C m-2 a-1
        res['C_tree'] = dat['bioTot']

        res['F_tree'] = -dat['NPP']

        # ----- Soil CO2, CH4 and N2O emissions
        vol = res['Vol'].values
        #F_soil = soilCO2(x=vol, ftype=ftype)
        res['F_soil'] = soilCO2(x=vol, ftype=ftype, wtd_para=wtd_para)
        res['C_soil'] = -np.cumsum(res['F_soil'].values) # change in soil C storage
        
        res['F_CH4'] = Fsoil[ftype]['CH4']
        res['F_N2O'] = Fsoil[ftype]['N2O']

        # ---- wood product pools and fluxes
        WPlong = wood_pools(S0=0.0, tau=tau_wpl)
        WPshort = wood_pools(S0=0.0, tau=tau_wps)

        Flong0 = alpha_saw * dat['harvest_Log'].values
        Fshort0 = (1 - alpha_saw) * dat['harvest_Log'].values + dat['harvest_Fibre'].values
 
        Sl = np.zeros(N)
        Ss = np.zeros(N)
        Fwpl = np.zeros(N)
        Fwps = np.zeros(N)
        
        for j in range(0,N):
            Sl[j], Fwpl[j] = WPlong.update(Flong0[j])
            Ss[j], Fwps[j] = WPshort.update(Fshort0[j])
            
        res['C_WP_long'] = Sl
        res['C_WP_short'] = Ss
        res['F_WP_long'] = Fwpl
        res['F_WP_short'] = Fwps       
        
        # fig, ax = plt.subplots(2,2)
        # ax[0,0].plot(res['C_WP_long'], 'o')
        # ax[0,1].plot(Fwpl, 'o')
        # ax[1,0].plot(Ss, 'o')
        # ax[1,1].plot(Fwps, 'o')       
        
        del Sl, Ss, Fwpl, Fwps

        # ---- residuepools and fluxes
        Rfol = wood_pools(S0=0.0, tau=tau_fol)
        Rfwd = wood_pools(S0=0.0, tau=tau_fwd)
        Rcwd = wood_pools(S0=0.0, tau=tau_cwd)

        S1 = np.zeros(N)
        S2 = np.zeros(N)
        S3 = np.zeros(N)
        F1 = np.zeros(N)
        F2 = np.zeros(N)
        F3 = np.zeros(N)
        
        ffol = dat['FFol'].values
        ffwd = dat['FWD'].values
        fcwd = dat['CWD'].values

        for j in range(0,N):
            S1[j], F1[j] = Rfol.update(ffol[j])
            S2[j], F2[j] = Rfwd.update(ffwd[j])
            S3[j], F3[j] = Rcwd.update(fcwd[j])

        res['C_resid'] = S1 + S2 + S3
        res['F_resid'] = F1 + F1 + F3
 
        # fig, ax = plt.subplots(2,2)
        # ax[0,0].plot(S1, 'o')
        # ax[0,1].plot(F1, 'o')
        # ax[1,0].plot(S2, 'o')
        # ax[1,1].plot(F2, 'o')    
        del S1, S2, S3, F1, F2, F3

    # restoration to open mires: whole stand is cut at t=0
    if ftype in ['RME', 'ROL', 'ROM']:

        res['year']  = dat['year']
        
        # ----- Soil CO2, CH4 and N2O emissions
        res['F_soil'] = Fsoil[ftype]['CO2']
        res['F_CH4'] = Fsoil[ftype]['CH4']
        res['F_N2O'] = Fsoil[ftype]['N2O']

        res['C_soil'] = -np.cumsum(res['F_soil'].values) # change in soil C storage

        # ---- wood product pools and fluxes
        WPlong = wood_pools(S0=0.0, tau=tau_wpl)
        WPshort = wood_pools(S0=0.0, tau=tau_wps)

        Flong0 = alpha_saw * dat['harvest_Log'].values
        Fshort0 = (1 - alpha_saw) * dat['harvest_Log'].values + dat['harvest_Fibre'].values
        
        Flong0[1:] = 0.0
        Fshort0[1:] = 0.0

        Sl = np.zeros(N)
        Ss = np.zeros(N)
        Fwpl = np.zeros(N)
        Fwps = np.zeros(N)
        
        for j in range(0,N):
            Sl[j], Fwpl[j] = WPlong.update(Flong0[j])
            Ss[j], Fwps[j] = WPshort.update(Fshort0[j])
            
        res['C_WP_long'] = Sl
        res['C_WP_short'] = Ss
        res['F_WP_long'] = Fwpl
        res['F_WP_short'] = Fwps       
    
        del Sl, Ss, Fwpl, Fwps

        # ---- residuepools and fluxes
        Rfol = wood_pools(S0=0.0, tau=tau_fol)
        Rfwd = wood_pools(S0=0.0, tau=tau_fwd)
        Rcwd = wood_pools(S0=0.0, tau=tau_cwdr) # coarse residues decompose slowly in aerobic conditions

        S1 = np.zeros(N)
        S2 = np.zeros(N)
        S3 = np.zeros(N)
        F1 = np.zeros(N)
        F2 = np.zeros(N)
        F3 = np.zeros(N)
        
        ffol = dat['FFol'].values; ffol[1:] = 0.0
        ffwd = dat['FWD'].values; ffwd[1:] = 0.0
        fcwd = dat['CWD'].values; fcwd[1:] = 0.0

        for j in range(0,N):
            S1[j], F1[j] = Rfol.update(ffol[j])
            S2[j], F2[j] = Rfwd.update(ffwd[j])
            S3[j], F3[j] = Rcwd.update(fcwd[j])

        res['C_resid'] = S1 + S2 + S3
        res['F_resid'] = F1 + F1 + F3
 
        del S1, S2, S3, F1, F2, F3

    # restoration to tree-covered mires
    if ftype in ['RSM', 'RPM']:
        print('restoring to open!')
        res[['Vol', 'BA']]  = dat[['vol', 'BA']].iloc[0].values
        res['year'] = dat['year']
        res['Age'] = dat['DomAge'].iloc[0] + np.arange(0,N)
        
        # all in g C m-2 or g C m-2 a-1
        res['C_tree'] = dat['bioTot'].iloc[0]

        # ----- Soil CO2, CH4 and N2O emissions
        res['F_soil'] = Fsoil[ftype]['CO2']
        res['F_CH4'] = Fsoil[ftype]['CH4']
        res['F_N2O'] = Fsoil[ftype]['N2O']

        res['C_soil'] = -np.cumsum(res['F_soil'].values) # change in soil C storage

    return res

# --- estimate radiative forcings

def compute_RFs(x, fyear=2025):
    cf = 1e-12 #1 g gas as Mton gas
    
    x = x.copy()
    x.index = x.index + fyear
    N = len(x)
    
    cols = ['RF_tot', 'RF_totCO2', 'RF_tree', 'RF_resid', 'RF_soil', 'RF_WP_short', 'RF_WP_long', 'RF_CH4', 'RF_N2O']
    # res = {'RF_tot': np.zeros(N), 'RF_totCO2': np.zeros(N), 'RF_tree': np.zeros(N), 'RF_resid': np.zeros(N), 'RF_soil': np.zeros(N), 
    #        'RF_WP_short': np.zeros(N), 'RF_WP_long': np.zeros(N), 
    #        'RF_CH4': np.zeros(N), 'RF_N2O': np.zeros(N)}
    res = pd.DataFrame(data = np.zeros((N,len(cols))), columns=cols, index=x.index - fyear)
    
    # CO2
    for c in ['F_tree', 'F_resid', 'F_soil', 'F_WP_short', 'F_WP_long']:
        fghg = pd.DataFrame(data=np.zeros((N, 3)), columns=['CO2', 'CH4', 'N2O'], index=x.index)
        fghg['CO2'] = x[c] * 44./12. * cf

        rf = radiative_forcing(fghg)
        #print(rf)
        res['R'+ c] = rf['CO2'].values
    
        del rf, fghg
    
    # CH4 and N2O
    fghg = pd.DataFrame(data=np.zeros((N, 3)), columns=['CO2', 'CH4', 'N2O'], index=x.index)
    fghg['CH4'] = x['F_CH4'] * cf
    fghg['N2O'] = x['F_N2O'] * cf    
    rf = radiative_forcing(fghg)

    res['RF_CH4'] = rf['CH4'].values
    res['RF_N2O'] = rf['N2O'].values
    
    # Total RF
    res['RF_tot'] = res['RF_tree'] + res['RF_resid'] + res['RF_soil'] + res['RF_WP_short'] + res['RF_WP_long'] + res['RF_CH4'] + res['RF_N2O']
    res['RF_totCO2'] = res['RF_tree'] + res['RF_resid'] + res['RF_soil'] + res['RF_WP_short'] + res['RF_WP_long']
    #res = pd.DataFrame.from_dict(res,)
    return res