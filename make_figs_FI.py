import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from stand_carbon import restoration_cases_new
from radiative_forcing import albedo_delta_rf

### ************ KUVIA SUOMENKIELISILLÄ TEKSTEILLÄ ***************

def make_fig4_FI(duration=200):
    # summarize results and compare with 'soil-only' from Laine et al. (2024)
    
    growth_scenarios = {'FNRsouth1': ['South', 'Rhtkg', {'CO2': 265.0, 'CH4': 0.34, 'N2O': 0.23}],
                        'FNRsouth2': ['South', 'Mtkg', {'CO2': 265.0, 'CH4': 0.34, 'N2O': 0.23}],
                        'FNRnorth1': ['North', 'Rhtkg', {'CO2': 265.0, 'CH4': 0.34, 'N2O': 0.23}],
                        'FNRnorth2': ['North', 'Mtkg', {'CO2': 265.0, 'CH4': 0.34, 'N2O': 0.23}],
                        'FNPsouth': ['South', 'Ptkg', {'CO2': -45.0, 'CH4': 0.34, 'N2O': 0.08}],
                        'FNPnorth': ['North', 'Ptkg', {'CO2': -45.0, 'CH4': 0.34, 'N2O': 0.08}],
                    }

    # prepare figure
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    tt = np.arange(0, duration, 1)

    #select 6 first colors from colormap
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color'][0:6]

    # plot
    y0 = 0.0
    y1 = 6e-14
    y2 = -6e-14
    
    ax[0].fill_between(tt, y0, y1, color='r', alpha=0.2)
    ax[0].fill_between(tt, y2, y0, color='b', alpha=0.2)
    ax[0].text(tt[4], 0.9*y1, 'a)')
    ax[0].set_ylim([y2, y1])
    ax[0].set_xlim([min(tt), max(tt)+1])
    ax[0].set_ylabel('$\overline{\Delta RF}_{[0,t]}$ (W m$^{-2}$ m$^{-2}$)')
    ax[0].set_xlabel('aika vettämisestä (vuosia)')
    
    #ax.plot(tt, np.zeros(len(tt)), 'k--', alpha=0.3)
    #ax[0].legend(fontsize=10, frameon=False, title=r'$t_0/t_{rot}$')
    ax[0].set_title('Rehvät turvekankaat')
    ax[0].text(tt[4], 0.9*y2, 'Ennallistaminen viilentää', fontsize=8)
    ax[0].text(tt[4], 0.8*y1, 'Ennallistaminen lämmittää', fontsize=8)

    ax[1].fill_between(tt, y0, y1, color='r', alpha=0.2)
    ax[1].fill_between(tt, y2, y0, color='b', alpha=0.2)
    ax[1].text(tt[4], 0.9*y1, 'b)')
    ax[1].set_ylim([y2, y1])
    ax[1].set_xlim([min(tt), max(tt)+1])
    ax[1].set_ylabel('$\overline{\Delta RF}_{[0,t]}$ (W m$^{-2}$ m$^{-2}$)')
    ax[1].set_xlabel('aika vettämisestä (vuosia)')
    
    #ax.plot(tt, np.zeros(len(tt)), 'k--', alpha=0.3)
    #ax[0].legend(fontsize=10, frameon=False, title=r'$t_0/t_{rot}$')
    ax[1].set_title('Karut turvekankaat')
    ax[1].text(tt[4], 0.9*y2, 'Ennallistaminen viilentää', fontsize=8)
    ax[1].text(tt[4], 0.8*y1, 'Ennallistaminen lämmittää', fontsize=8)

    
    #--- Fertile peatland restoration pathways---

    # rtypenames1 = ['Spruce mire', 'open eu/mesotrophic', 'open oligotrophic']
    # rtypes1 = ['RSM', 'RME', 'ROL']

    # ftypes1 = ['FNRsouth1', 'FNRsouth2', 'FNRnorth1', 'FNRnorth2']

    rtypenames1 = ['Korpi', 'ruohoinen avosuo', ' sarainen avosuo']
    rtypes1 = ['RSM', 'RME', 'ROL']

    ftypes1 = ['FNRsouth1', 'FNRsouth2', 'FNRnorth1', 'FNRnorth2']

    n = len(ftypes1)
    
    tmp1 = np.ones((n, duration))*np.NaN # biome scale
    #tmp2 = np.ones((n, duration))*np.NaN # soil only
    tmp2 = np.ones((duration))*np.NaN # L24

    j = 0
    for rt in rtypes1:
        k = 0
        for ft in ftypes1:
            RF1, RFr1, S1, Sr1, Fco21, Frco21, Sattr1  = restoration_cases_new(ftype=ft, rtype=rt, duration=200, start_yr=2025)

            # -- radiative forcings

            # site-level
            if rt in ['RSM', 'RPM']:# restoring without clear-cutting
                rfeco = RF1['Soil'] + RF1['Tree'] + RF1['Res']
                rfreco = RFr1['Soil']
                delta_rfeco = rfreco['Tot'] - rfeco['Tot'] 
            else:
                rfeco = RF1['Soil'] + RF1['Tree'] + RF1['Res']
                rfreco = RFr1['Soil'] +  RFr1['Res']
                delta_rfeco = rfreco['Tot'] - rfeco['Tot'] + RF1['deltaRFa']  
     
            #biome level, incl. wood use
            if rt in ['RSM', 'RPM']: # restoring without clear-cutting
                rfbio = RF1['Soil'] + RF1['Tree'] + RF1['WP'] + RF1['Res']
                rfrbio = RFr1['Soil']
                delta_rfbio = rfrbio['Tot'] - rfbio['Tot']

                # albedo effect
                rfalb = RF1['deltaRFa'] # change in albedo RF in forestry relative to open peatland
                aa = max(Sattr1['Vol'].values)*np.ones(len(rfalb))
                rrfalb = albedo_delta_rf(aa) # change in RF of mature restored stand to open peatland
                delta_rf_alb = -1* (rrfalb - rfalb) # difference
                delta_rfbio = delta_rfbio + delta_rf_alb
            else:
                rfbio = RF1['Soil'] + RF1['Tree'] + RF1['WP'] + RF1['Res']
                rfrbio = RFr1['Soil'] + RFr1['WP']+ RFr1['Res']
                delta_rfbio = rfrbio['Tot'] - rfbio['Tot'] + RF1['deltaRFa']

            # soil only
            rfsoil = RF1['Soil'] 
            rfrsoil = RFr1['Soil']
            delta_rfsoil = rfrsoil['Tot'] - rfsoil['Tot']

            #laine 2024
            delta_rfL24 = RFr1['Soil']['Tot'] - RF1['L24']['Tot']

            tmp1[k,:] = np.cumsum(delta_rfbio.values) / (tt +1)
            tmp2[:] = np.cumsum(delta_rfL24.values) / (tt +1)

            k += 1

        # compute mean & range
        y1 = np.mean(tmp1, axis=0)
        y1_l = np.min(tmp1, axis=0)
        y1_h = np.max(tmp1, axis=0)

        ax[0].fill_between(tt, y1_l, y1_h, color=colors[j], alpha=0.3)
        ax[0].plot(tt, y1, '-', color=colors[j], linewidth=2, label=rtypenames1[j])
        ax[0].plot(tt, tmp2, '--', color=colors[j], linewidth=1)

        j += 1
    
    #ax[0].text(tt[4], 0.9*y1, 'a)') 
    ax[0].legend(fontsize=8, frameon=False, loc=1)

    #--- Nutrient poor peatland restoration pathways

    rtypenames2 = ['Räme', 'sarainen avosuo']#, 'open ombotrophic']
    rtypes2 = ['RPM', 'ROL']#, 'ROM']

    ftypes2 = ['FNPsouth', 'FNPnorth']
    
    n = len(ftypes2)
    
    tmp1 = np.ones((n, duration))*np.NaN # biome scale
    tmp2 = np.ones((duration))*np.NaN # L24

    j = 0
    for rt in rtypes2:
        k = 0
        for ft in ftypes2:
            RF1, RFr1, S1, Sr1, Fco21, Frco21, Sattr1  = restoration_cases_new(ftype=ft, rtype=rt, duration=200, start_yr=2025)

            # -- radiative forcings

            # site-level
            if rt in ['RSM', 'RPM']:# restoring without clear-cutting
                rfeco = RF1['Soil'] + RF1['Tree'] + RF1['Res']
                rfreco = RFr1['Soil']
                delta_rfeco = rfreco['Tot'] - rfeco['Tot'] 
            else:
                rfeco = RF1['Soil'] + RF1['Tree'] + RF1['Res']
                rfreco = RFr1['Soil'] +  RFr1['Res']
                delta_rfeco = rfreco['Tot'] - rfeco['Tot'] + RF1['deltaRFa']  
     
            #biome level, incl. wood use
            if rt in ['RSM', 'RPM']: # restoring without clear-cutting
                rfbio = RF1['Soil'] + RF1['Tree'] + RF1['WP'] + RF1['Res']
                rfrbio = RFr1['Soil']
                delta_rfbio = rfrbio['Tot'] - rfbio['Tot']
                
                # albedo effect
                rfalb = RF1['deltaRFa'] # change in albedo RF in forestry relative to open peatland
                aa = max(Sattr1['Vol'].values)*np.ones(len(rfalb))
                rrfalb = albedo_delta_rf(aa) # change in RF of mature restored stand to open peatland
                delta_rf_alb = -1* (rrfalb - rfalb) # difference
                delta_rfbio = delta_rfbio + delta_rf_alb

            else:
                rfbio = RF1['Soil'] + RF1['Tree'] + RF1['WP'] + RF1['Res']
                rfrbio = RFr1['Soil'] + RFr1['WP']+ RFr1['Res']
                delta_rfbio = rfrbio['Tot'] - rfbio['Tot'] + RF1['deltaRFa']

        # soil only
            rfsoil = RF1['Soil'] 
            rfrsoil = RFr1['Soil']
            delta_rfsoil = rfrsoil['Tot'] - rfsoil['Tot']

            #laine 2024
            delta_rfL24 = RFr1['Soil']['Tot'] - RF1['L24']['Tot']

            tmp1[k,:] = np.cumsum(delta_rfbio.values) / (tt +1)
            tmp2[:] = np.cumsum(delta_rfL24.values) / (tt +1)

            k += 1

        
        # compute mean & range
        y1 = np.mean(tmp1, axis=0)
        y1_l = np.min(tmp1, axis=0)
        y1_h = np.max(tmp1, axis=0)

        ax[1].fill_between(tt, y1_l, y1_h, color=colors[j+3], alpha=0.3)
        ax[1].plot(tt, y1, '-', color=colors[j+3], linewidth=2, label=rtypenames2[j])
        ax[1].plot(tt, tmp2, '--', color=colors[j+3], linewidth=1)

        j += 1
    #ax[1].text(tt[4], 0.9*y1, 'b)')    
    ax[1].legend(fontsize=8, frameon=False, loc=4)
    
    return fig

def make_fig2_FI(ftype1, rtype1, ftype2, rtype2, duration, ylims=[-1.e-13, 1.1e-13], savefig=False):

    growth_scenarios = {'FNRsouth1': ['South', 'Rhtkg', {'CO2': 265.0, 'CH4': 0.34, 'N2O': 0.23}],
                        'FNRsouth2': ['South', 'Mtkg', {'CO2': 265.0, 'CH4': 0.34, 'N2O': 0.23}],
                        'FNRnorth1': ['North', 'Rhtkg', {'CO2': 265.0, 'CH4': 0.34, 'N2O': 0.23}],
                        'FNRnorth2': ['North', 'Mtkg', {'CO2': 265.0, 'CH4': 0.34, 'N2O': 0.23}],
                        'FNPsouth': ['South', 'Ptkg', {'CO2': -45.0, 'CH4': 0.34, 'N2O': 0.08}],
                        'FNPnorth': ['North', 'Ptkg', {'CO2': -45.0, 'CH4': 0.34, 'N2O': 0.08}],
                    }
    
    v = growth_scenarios[ftype1]
    ss = ftype1[0:3]

    ttext1 = 'Rehevä turvekangas --> Ruohoinen avosuo.'

    v = growth_scenarios[ftype2]
    ss2 = ftype2[0:3]
    ttext2 = 'Rehevä turvekangas --> Korpi.'

    # Case 1: restore FNR --> open RME
    RF1, RFr1, S1, Sr1, Fco21, Frco21, Sattr1  = restoration_cases_new(ftype=ftype1, rtype=rtype1, duration=duration, start_yr=2025)

    # -- radiative forcings

    # site-level
    rfeco = RF1['Soil'] + RF1['Tree'] + RF1['Res']
    rfreco = RFr1['Soil'] +  RFr1['Res']
    delta_rfeco = rfreco - rfeco

    #biome level, incl. wood use
    rfbio = RF1['Soil'] + RF1['Tree'] + RF1['WP'] + RF1['Res']
    rfrbio = RFr1['Soil'] + RFr1['WP']+ RFr1['Res']
    delta_rfbio = rfrbio - rfbio

    # soil only
    rfsoil = RF1['Soil'] 
    rfrsoil = RFr1['Soil']
    delta_rfsoil = rfrsoil - rfsoil

    #laine 2024
    delta_rfL24 = RFr1['Soil'] - RF1['L24']

    # --- fluxes
    # forest
    nee = (Fco21['Tree'] + Fco21['Soil'] + Fco21['Res']) * 12./44
    npb = nee + Fco21['Res'].values*12/44
    #restored
    neer = Frco21['Soil']*12/44 + Frco21['Res']*12/44
    npbr = neer + Frco21['Res'].values*12/44

    # --- storages
    stot = S1['Tree']  + S1['Soil']  + S1['Res'] + S1['WP']
    srtot = Sr1['Soil']# + Sr1['Res']

    # Case 2: restoring without clear-cutting

    RF2, RFr2, S2, Sr2, Fco22, Frco22, Sattr2  = restoration_cases_new(ftype=ftype2, rtype=rtype2, duration=duration, start_yr=2025)
    # -- radiative forcings

    # albedo effect

    rfalb = RF2['deltaRFa'] # change in albedo RF in forestry relative to open peatland
    aa = max(Sattr2['Vol'].values)*np.ones(len(rfalb))
    
    rrfalb = albedo_delta_rf(aa) # change in RF of mature restored stand to open peatland

    delta_rf_alb = -1* (rrfalb - rfalb)

    # site-level
    rfeco = RF2['Soil'] + RF2['Tree'] + RF2['Res']
    rfreco = RFr2['Soil']
    delta_rfeco2 = rfreco - rfeco

    #biome level, incl. wood use
    rfbio = RF2['Soil'] + RF2['Tree'] + RF2['WP'] + RF2['Res']
    rfrbio = RFr2['Soil']
    delta_rfbio2 = rfrbio - rfbio

    # soil only
    rfsoil = RF2['Soil'] 
    rfrsoil = RFr2['Soil']
    delta_rfsoil2 = rfrsoil - rfsoil

    #laine 2024
    delta_rfL242 = RFr2['Soil'] - RF2['L24']

    ## --- plot figure

    rco2 = 44./12

    fig, ax = plt.subplots(2,2,figsize=(12,10))

    tt = S1['Tree'].index.values
    tt = tt - min(tt)

    # radiative forcings
    y0 = 0.0

    y1 = 1.1*np.max([np.abs(delta_rfsoil['Tot'].values), np.abs(delta_rfbio['Tot'].values), np.abs(delta_rfeco['Tot'].values)])
    y2 = 1.1*np.min([delta_rfsoil['Tot'].values, np.abs(delta_rfbio['Tot'].values)])
    y1 = 1.5e-13
    y2 = -1.5e-13
    ax[0,0].fill_between(tt, y0, y1, color='r', alpha=0.2)
    ax[0,0].fill_between(tt, y2, y0, color='b', alpha=0.2)
    ax[0,0].text(tt[4], 0.9*ylims[1], 'a)')

    ax[0,0].set_title(ttext1)
    ax[0,0].plot(tt, delta_rfbio['Tot'].values + RF1['deltaRFa'],'-', linewidth=2, label='Tot.')

    ax[0,0].plot(tt, delta_rfeco['Tot'].values + RF1['deltaRFa'], '--', linewidth=1.5, label='Tot. (ilman puutuotteita)')
    ax[0,0].plot(tt, delta_rfsoil['Tot'], '-', linewidth=1.5, label='Maaperä')

    ax[0,0].plot(tt, delta_rfL24['Tot'], 'k-',linewidth=1.0, alpha=0.5, label='Tot. (Laine et al. 2024)')

    ax[0,0].legend(fontsize=8, frameon=False, loc=3)
    ax[0,0].set_ylabel('$\Delta RF$ (W m$^{-2}$(earth) m$^{-2}$)')

    ax[0,0].set_xlim([min(tt), max(tt)+1])
    ax[0,0].set_ylim(ylims[0], ylims[1])

    # contributions
    ax[0,1].fill_between(tt, y0, y1, color='r', alpha=0.2)
    ax[0,1].fill_between(tt, y2, y0, color='b', alpha=0.2)
    ax[0,1].text(tt[4], 0.9*ylims[1], 'b)')

    ax[0,1].set_title('Eri tekijöiden vaikutus')

    ax[0,1].plot(tt, delta_rfbio['Tot'].values + RF1['deltaRFa'],'-', linewidth=2, label='Tot.')
    ax[0,1].plot(tt, delta_rfbio['CO2'].values,'--', linewidth=1.5, label='CO$_2$')
    ax[0,1].plot(tt, delta_rfbio['CH4'].values, '--', linewidth=1.5, label='CH$_4$')
    ax[0,1].plot(tt, delta_rfbio['N2O'].values, '--', linewidth=1.5, label='N$_20$')
    ax[0,1].plot(tt, RF1['deltaRFa'], 'k-', linewidth=1.0, alpha=0.5, label='Albedo')

    ax[0,1].legend(fontsize=8, frameon=False)
    ax[0,1].set_ylabel('$\Delta RF$ (W m$^{-2}$ m$^{-2}$)')

    ax[0,1].set_xlim([min(tt), max(tt)+1])
    ax[0,1].set_ylim(ylims[0], ylims[1])
    ax[0,1].text(tt[4], 0.85*ylims[0], 'Ennallistaminen viilentää', fontsize=8)
    ax[0,1].text(tt[4], 0.8*ylims[1], 'Ennallistaminen lämmittää', fontsize=8)

    # radiative forcings
    y0 = 0.0

    y1 = 1.1*np.max([np.abs(delta_rfsoil2['Tot'].values), np.abs(delta_rfbio2['Tot'].values), np.abs(delta_rfeco2['Tot'].values)])
    y2 = 1.1*np.min([delta_rfsoil2['Tot'].values, np.abs(delta_rfbio2['Tot'].values)])
    y1 = 1.5e-13
    y2 = -1.5e-13
    ax[1,0].fill_between(tt, y0, y1, color='r', alpha=0.2)
    ax[1,0].fill_between(tt, y2, y0, color='b', alpha=0.2)
    ax[1,0].text(tt[4], 0.9*ylims[1], 'c)')
    ax[1,0].set_title(ttext2)

    ax[1,0].plot(tt, delta_rfbio2['Tot'].values + delta_rf_alb,'-', linewidth=2, label='Tot.')

    ax[1,0].plot(tt, delta_rfeco2['Tot'].values + delta_rf_alb, '--', linewidth=1.5, label='Tot. (ilman puutuotteita)')
    ax[1,0].plot(tt, delta_rfsoil2['Tot'], '-', linewidth=1.5, label='Maaperä')
    ax[1,0].plot(tt, delta_rfL242['Tot'], 'k-',linewidth=1.0, alpha=0.5, label='Tot. (Laine et al. 2024)')

    ax[1,0].legend(fontsize=8, frameon=False, loc=3)
    ax[1,0].set_ylabel('$\Delta RF$ (W m$^{-2}$ m$^{-2}$)')
    ax[1,0].set_xlabel('aika vettämisestä (vuosia)')
    ax[1,0].set_xlim([min(tt), max(tt)+1])
    ax[1,0].set_ylim(ylims[0], ylims[1])

    # contributions
    ax[1,1].fill_between(tt, y0, y1, color='r', alpha=0.2)
    ax[1,1].fill_between(tt, y2, y0, color='b', alpha=0.2)
    ax[1,1].text(tt[4], 0.9*ylims[1], 'd)')

    ax[1,1].set_title('Eri tekijöiden vaikutus')
    #ax[0,1].plot(tt, np.zeros(len(tt)), 'k--', alpha=0.5)
    ax[1,1].plot(tt, delta_rfbio2['Tot'].values + delta_rf_alb,'-', linewidth=2, label='Total')
    ax[1,1].plot(tt, delta_rfbio2['CO2'].values,'--', linewidth=1.5, label='CO$_2$')
    ax[1,1].plot(tt, delta_rfbio2['CH4'].values, '--', linewidth=1.5, label='CH$_4$')
    ax[1,1].plot(tt, delta_rfbio2['N2O'].values, '--', linewidth=1.5, label='N$_20$')
    ax[1,1].plot(tt, delta_rf_alb, 'k-',linewidth=1.0, alpha=0.5, label='Albedo')

    ax[1,1].legend(fontsize=8, frameon=False)
    ax[1,1].set_ylabel('$\Delta RF$ (W m$^{-2}$ m$^{-2}$)')
    ax[1,1].set_xlabel('aika vettämisestä (vuosia)')
    ax[1,1].set_xlim([min(tt), max(tt)+1])
    ax[1,1].set_ylim(ylims[0], ylims[1])
    ax[1,1].text(tt[4], 0.85*ylims[0], 'Ennallistaminen viilentää', fontsize=8)
    ax[1,1].text(tt[4], 0.8*ylims[1], 'Ennallistaminen lämmittää', fontsize=8)


    return fig

def make_fig3_FI(ftype, rtype, t_rot, duration=200, v='Tot'):
    
#Case 3: Does it matter when we restore? Restore FNR --> open RME at different parts of rotation cycle

    dt = np.fix(np.array([1.0, 0.8, 0.6, 0.4, 0.2])*t_rot)

    tms = 2025 + dt

    n = len(tms)
    res = {'t0': np.ones(n)*np.NaN, 'deltaRFbio': np.ones((n, duration))*np.NaN, 'deltaRFeco': np.ones((n, duration))*np.NaN, 'deltaRFsoil': np.ones((n, duration))*np.NaN,
        'Stree': np.ones((n, duration))*np.NaN, 'Ssoil': np.ones((n, duration))*np.NaN, 'Sres': np.ones((n, duration))*np.NaN, 'Swp': np.ones((n, duration))*np.NaN,
        'Srsoil': np.ones((n, duration))*np.NaN, 'Srres': np.ones((n, duration))*np.NaN, 'Srwp': np.ones((n, duration))*np.NaN, 't_rot': np.ones((n))*np.NaN,
        'deltaRFa': np.ones((n, duration)), 'Frco2': np.ones((n, duration)), 'Fco2': np.ones((n, duration))
        }

    k = 0
    for t0 in tms:
        #print(t0)
        RF, RFr, S, Sr, Fco2, Frco2, Sattr = restoration_cases_new(ftype=ftype, rtype=rtype, duration=duration, start_yr=t0)
        #RF, RFr, S, Sr, Fco2, Frco2, Sattr = restoration_cases_new(ftype='FNRnorth2', rtype='RME', duration=duration, start_yr=t0)

        res['t0'][k] = t0
        res['Stree'][k,:] = S['Tree'].values
        res['Ssoil'][k,:] = S['Soil'].values
        res['Sres'][k,:] = S['Res']
        res['Swp'][k,:] = S['WP']
        res['Fco2'][k,:] = Fco2['Tree'] + Fco2['Soil'] + Fco2['Res'] + Fco2['WP']
        res['Srsoil'][k,:] = Sr['Soil'].values
        res['Srres'][k,:] = Sr['Res']
        res['Srwp'][k,:] = Sr['WP']
        res['Frco2'][k,:] = Frco2['Soil'] + Frco2['Res'] + Frco2['WP']

        #res['Vol'][k] = Sattr['Vol'].values[0]

        # site-level
        rfeco = RF['Soil'] + RF['Tree'] + RF['Res']
        rfreco = RFr['Soil'] +  RFr['Res']
        delta_rfeco = rfreco - rfeco

        delta_rfsoil = RFr['Soil']- RF['Soil']

        #biome level, incl. wood use
        rfbio = RF['Soil'] + RF['Tree'] + RF['WP'] + RF['Res']
        rfrbio = RFr['Soil'] + RFr['WP']+ RFr['Res']
        delta_rfbio = rfrbio - rfbio

        if v == 'Tot':
            res['deltaRFeco'][k,:] = delta_rfeco['Tot'].values.reshape((duration,))
            res['deltaRFbio'][k,:] = delta_rfbio['Tot'].values.reshape((duration,)) + RF['deltaRFa'].reshape((duration,))
        else:
            res['deltaRFeco'][k,:] = delta_rfeco[v].values.reshape((duration,))
            res['deltaRFbio'][k,:] = delta_rfbio[v].values.reshape((duration,))

        res['deltaRFa'][k,:] = RF['deltaRFa'].reshape((duration,))
        res['deltaRFsoil'][k,:] =  delta_rfsoil['CO2'].values.reshape((duration,))
        
        k +=1


    # -----  plot figs ---------

    fig, ax = plt.subplots(1,1, figsize=(6,5))
    
    tt = np.arange(0, duration, 1)
    #n = len(res)
    for k in range(n):
        #print(k, res['t0'][k])
        st = (res['t0'][k] - 2025) / t_rot
        if st <= 0.5:
            ls = '--'
        else:
            ls = '-'
        #st = 1-((res['t0'][k] - 2025)/t_rot)
        st = (res['t0'][k] - 2025) / t_rot
        #print(st)
        aa = np.cumsum(res['deltaRFbio'][k,:]) / (tt +1)
        #aa = res['deltaRFbio'][k,:] + res['deltaRFa'][k,:]
        if k==0:
            lw = 2
        else:
            lw = 1.0
        ax.plot(tt, aa, linestyle=ls, linewidth=lw, label='%.1f' %st)

    y0 = 0.0
    y1 = 4e-14
    y2 = -2e-14
    ax.fill_between(tt, y0, y1, color='r', alpha=0.2)
    ax.fill_between(tt, y2, y0, color='b', alpha=0.2)
    ax.set_ylim([y2, y1])
    ax.text(tt[4], 0.9*y1, 'a)')
    ax.set_xlim([min(tt), max(tt)+1])
    ax.set_ylabel('$\overline{\Delta RF}_{[0,t]}$ (W m$^{-2}$ m$^{-2}$)')
    ax.set_xlabel('aika vettämisestä (vuosia)')
    #ax.plot(tt, np.zeros(len(tt)), 'k--', alpha=0.3)
    ax.legend(fontsize=10, frameon=False, title=r'$t_0/t_{rot}$')
    ax.set_title('Rehevä turvekangas --> ruohoinen avosuo.\n Vettäminen kiertoajan ($t_{rot}$) eri vaiheissa.')
    ax.text(tt[4], 0.9*y2, 'Ennallistaminen viilentää', fontsize=8)
    ax.text(tt[4], 0.8*y1, 'Ennallistaminen lämmittää', fontsize=8)

    
    return fig