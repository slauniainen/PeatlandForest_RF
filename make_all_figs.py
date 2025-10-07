import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from stand_carbon import restoration_cases_new
from radiative_forcing import albedo_delta_rf, radiative_forcing



def make_fig2(ftype1, rtype1, ftype2, rtype2, duration, ylims=[-1.1e-13, 1.1e-13], savefig=False):

    growth_scenarios = {'FNRsouth1': ['South', 'Rhtkg', {'CO2': 265.0, 'CH4': 0.34, 'N2O': 0.23}],
                        'FNRsouth2': ['South', 'Mtkg', {'CO2': 265.0, 'CH4': 0.34, 'N2O': 0.23}],
                        'FNRnorth1': ['North', 'Rhtkg', {'CO2': 265.0, 'CH4': 0.34, 'N2O': 0.23}],
                        'FNRnorth2': ['North', 'Mtkg', {'CO2': 265.0, 'CH4': 0.34, 'N2O': 0.23}],
                        'FNPsouth': ['South', 'Ptkg', {'CO2': -45.0, 'CH4': 0.34, 'N2O': 0.08}],
                        'FNPnorth': ['North', 'Ptkg', {'CO2': -45.0, 'CH4': 0.34, 'N2O': 0.08}],
                    }
    
    v = growth_scenarios[ftype1]
    ss = ftype1[0:3]
    ofile = r'MS/Figs/Case_1&2_%s_%s.png' %(ss, v[1] + ' ' + v[0])
    ofile2 = r'MS/Figs/Case_1&2_%s_%s_attributes.png' %(ss, v[1] + ' ' + v[0])

    ttext1 = 'Case 1: %s (%s %s) --> %s. Clear-cut at t=0.' %(ss, v[1], v[0], rtype1)
    ttext0 = 'Case 1: %s (%s). Change in C storage since t=0.' %(v[1], v[0])

    v = growth_scenarios[ftype2]
    ss2 = ftype2[0:3]
    ttext2 = 'Case 2: %s (%s %s) --> %s. No harvest at t=0.' %(ss2, v[1], v[0], rtype2)

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

    #---- albedo-effect is the difference!

    rfalb = RF2['deltaRFa'] # change in albedo RF in forestry relative to open peatland
    aa = max(Sattr2['Vol'].values)*np.ones(len(rfalb))
    
    rrfalb = albedo_delta_rf(aa) # change in RF of mature restored stand to open peatland

    delta_rf_alb = -1* (rrfalb - rfalb)

    # plt.figure()
    # tt = np.arange(0,len(delta_rf_alb))
    # plt.plot(tt, rfalb, 'r-')
    # plt.plot(tt, rrfalb, 'g-')
    # plt.plot(tt, delta_rf_alb, 'k-')
    
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
    fig2, ax2 = plt.subplots(1,2,figsize=(12,6))

    tt = S1['Tree'].index.values
    tt = tt - min(tt)

    stot = S1['Tree']  + S1['Soil']  + S1['Res'] + S1['WP']

    ax2[0].plot(tt, np.zeros(len(tt)), 'k--', alpha=0.5)
    ax2[0].plot(tt, stot, linewidth=2, label='Total')
    ax2[0].plot(tt, S1['Tree'], label='Stand')
    ax2[0].plot(tt, S1['Soil'], label='Soil')
    ax2[0].plot(tt, S1['Res'], '-', label='Res')
    ax2[0].plot(tt, S1['WP'], '-', label='WP')
    ax2[0].plot(tt, srtot, '--', linewidth=2, label='Soil\nrestored')
    ax2[0].legend(frameon=False)
    ax2[0].set_ylabel('$\Delta S_i$ (kg C m$^{-2}$)')
    ax2[0].set_xlabel('years')
    ax2[0].set_xlim([min(tt), max(tt)+1])
    ax2[0].set_title(ttext0)

    ax2[1].plot(tt, np.zeros(len(tt)), 'k--', alpha=0.5)
    ax2[1].plot(tt, npb, '-', linewidth=2, label='NEE + WP')
    ax2[1].plot(tt, nee, '-', linewidth=1, label='NEE')
    ax2[1].plot(tt, npbr, '-', linewidth=1, label='NEE + WP\nrestored')
    ax2[1].plot(tt, neer, '-', linewidth=1, label='NEE\nrestored')
    ax2[1].legend(frameon=True)
    ax2[1].set_ylabel('g C m$^{-2}$ a$^{-1}$')
    ax2[1].set_xlabel('years')
    ax2[1].set_xlim([min(tt), max(tt)+1])
    #ax[1,0].set_ylim(y2, y1)

    scen = growth_scenarios[ftype1]
    ax2[1].set_title(scen[0] + ', ' + scen[1] + ' Net CO$_2$ flux.')

    # radiative forcings
    y0 = 0.0

    y1 = 1.1*np.max([np.abs(delta_rfsoil['Tot'].values), np.abs(delta_rfbio['Tot'].values), np.abs(delta_rfeco['Tot'].values)])
    y2 = 1.1*np.min([delta_rfsoil['Tot'].values, np.abs(delta_rfbio['Tot'].values)])
    y1 = 1.5e-13
    y2 = -1.5e-13
    ax[0,0].fill_between(tt, y0, y1, color='r', alpha=0.2)
    ax[0,0].fill_between(tt, y2, y0, color='b', alpha=0.2)

    ax[0,0].set_title(ttext1)
    ax[0,0].plot(tt, delta_rfbio['Tot'].values + RF1['deltaRFa'],'-', linewidth=2, label='Total')

    ax[0,0].plot(tt, delta_rfeco['Tot'].values + RF1['deltaRFa'], '--', linewidth=1.5, label='Total\n(no WP)')
    ax[0,0].plot(tt, delta_rfsoil['Tot'], '-', linewidth=1.5, label='Soil')

    ax[0,0].plot(tt, delta_rfL24['Tot'], 'k-',linewidth=1.0, alpha=0.5, label='Total\nL24')

    ax[0,0].legend(fontsize=8, frameon=False)
    ax[0,0].set_ylabel('$\Delta RF$ (W m$^{-2}$(earth) m$^{-2}$ (land restored))')

    ax[0,0].set_xlim([min(tt), max(tt)+1])
    ax[0,0].set_ylim(ylims[0], ylims[1])

    #ax[0,0].text(tt[2], 0.9*y2, 'Restoration cooling', fontsize=8)
    #ax[0,0].text(tt[2], 0.9*y1, 'Restoration warming', fontsize=8)

    # contributions
    ax[0,1].fill_between(tt, y0, y1, color='r', alpha=0.2)
    ax[0,1].fill_between(tt, y2, y0, color='b', alpha=0.2)

    ax[0,1].set_title('Case 1: Contributions')
    #ax[0,1].plot(tt, np.zeros(len(tt)), 'k--', alpha=0.5)
    ax[0,1].plot(tt, delta_rfbio['Tot'].values + RF1['deltaRFa'],'-', linewidth=2, label='Total')
    ax[0,1].plot(tt, delta_rfbio['CO2'].values,'--', linewidth=1.5, label='CO$_2$')
    ax[0,1].plot(tt, delta_rfbio['CH4'].values, '--', linewidth=1.5, label='CH$_4$')
    ax[0,1].plot(tt, delta_rfbio['N2O'].values, '--', linewidth=1.5, label='N$_20$')
    ax[0,1].plot(tt, RF1['deltaRFa'], '--', linewidth=1.5, label='Albedo')
    #ax[0,1].plot(tt, delta_rfL24['Tot'], 'k-',linewidth=1.0, alpha=0.5, label='Total\nL24')

    #ax[0,1].plot(tt, RF1['deltaRFa'], '-', linewidth=1.0, label='Albedo')

    ax[0,1].legend(fontsize=8, frameon=False)
    ax[0,1].set_ylabel('$\Delta RF$ (W m$^{-2}$(earth) m$^{-2}$ (land restored))')
    #ax[0,1].set_xlabel('yr since restoration')
    ax[0,1].set_xlim([min(tt), max(tt)+1])
    ax[0,1].set_ylim(ylims[0], ylims[1])
    ax[0,1].text(tt[4], 0.9*ylims[0], 'Restoration cooling', fontsize=8)
    ax[0,1].text(tt[4], 0.9*ylims[1], 'Restoration warming', fontsize=8)

    # radiative forcings
    y0 = 0.0

    y1 = 1.1*np.max([np.abs(delta_rfsoil2['Tot'].values), np.abs(delta_rfbio2['Tot'].values), np.abs(delta_rfeco2['Tot'].values)])
    y2 = 1.1*np.min([delta_rfsoil2['Tot'].values, np.abs(delta_rfbio2['Tot'].values)])
    y1 = 1.5e-13
    y2 = -1.5e-13
    ax[1,0].fill_between(tt, y0, y1, color='r', alpha=0.2)
    ax[1,0].fill_between(tt, y2, y0, color='b', alpha=0.2)

    ax[1,0].set_title(ttext2)

    ax[1,0].plot(tt, delta_rfbio2['Tot'].values + delta_rf_alb,'-', linewidth=2, label='Total')

    ax[1,0].plot(tt, delta_rfeco2['Tot'].values + delta_rf_alb, '--', linewidth=1.5, label='Total\n(no WP)')
    ax[1,0].plot(tt, delta_rfsoil2['Tot'], '-', linewidth=1.5, label='Soil')
    ax[1,0].plot(tt, delta_rfL242['Tot'], 'k-',linewidth=1.0, alpha=0.5, label='Total\nL24')

    #ax[0,0].plot(tt, delta_rfL24['Tot'], 'k-',linewidth=1.0, alpha=0.5, label='L24')

    ax[1,0].legend(fontsize=8, frameon=False)
    ax[1,0].set_ylabel('$\Delta RF$ (W m$^{-2}$(earth) m$^{-2}$ (land restored))')
    ax[1,0].set_xlabel('year since restoration')
    ax[1,0].set_xlim([min(tt), max(tt)+1])
    ax[1,0].set_ylim(ylims[0], ylims[1])
    #ax[0,0].text(tt[2], 0.9*y2, 'Restoration cooling', fontsize=8)
    #ax[0,0].text(tt[2], 0.9*y1, 'Restoration warming', fontsize=8)

    # contributions
    ax[1,1].fill_between(tt, y0, y1, color='r', alpha=0.2)
    ax[1,1].fill_between(tt, y2, y0, color='b', alpha=0.2)

    ax[1,1].set_title('Case 2: Contributions')
    #ax[0,1].plot(tt, np.zeros(len(tt)), 'k--', alpha=0.5)
    ax[1,1].plot(tt, delta_rfbio2['Tot'].values + delta_rf_alb, '-', linewidth=2, label='Total')
    ax[1,1].plot(tt, delta_rfbio2['CO2'].values,'--', linewidth=1.5, label='CO$_2$')
    ax[1,1].plot(tt, delta_rfbio2['CH4'].values, '--', linewidth=1.5, label='CH$_4$')
    ax[1,1].plot(tt, delta_rfbio2['N2O'].values, '--', linewidth=1.5, label='N$_20$')
    ax[1,1].plot(tt, delta_rf_alb, '--', linewidth=1.5, label='Albedo')
    #ax[1,1].plot(tt, delta_rfL242['Tot'], 'k-',linewidth=1.0, alpha=0.5, label='Total\nL24')

    #ax[0,1].plot(tt, RF1['deltaRFa'], '-', linewidth=1.0, label='Albedo')

    ax[1,1].legend(fontsize=8, frameon=False)
    ax[1,1].set_ylabel('$\Delta RF$ (W m$^{-2}$(earth) m$^{-2}$ (land restored))')
    ax[1,1].set_xlabel('year since restoration')
    ax[1,1].set_xlim([min(tt), max(tt)+1])
    ax[1,1].set_ylim(ylims[0], ylims[1])
    ax[1,1].text(tt[4], 0.9*ylims[0], 'Restoration cooling', fontsize=8)
    ax[1,1].text(tt[4], 0.9*ylims[1], 'Restoration warming', fontsize=8)

    if savefig == True:
        fig.savefig(ofile, dpi=600)
        fig2.savefig(ofile2, dpi=600)
    return fig, fig2

def make_fig3(ftype, rtype, t_rot, duration=200, v='Tot'):
    
#Case 3: Does it matter when we restore? Restore FNR --> open RME at different parts of rotation cycle

    #duration = 200

    #dt = np.fix(np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9])*t_rot)
    #dt = np.fix(np.array([1.0, 0.9, 0.7, 0.5, 0.3, 0.1])*t_rot)
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

    # Case 4: linear transition of soil fluxes

    #tms = np.arange(0,45, 5)
    tms = [0, 5, 10, 20, 40]
    n4 = len(tms)
    res4 = {'t0': np.ones(n)*np.NaN, 'deltaRFbio': np.ones((n, duration))*np.NaN, 'deltaRFeco': np.ones((n, duration))*np.NaN, 'deltaRFsoil': np.ones((n, duration))*np.NaN,
        'Stree': np.ones((n, duration))*np.NaN, 'Ssoil': np.ones((n, duration))*np.NaN, 'Sres': np.ones((n, duration))*np.NaN, 'Swp': np.ones((n, duration))*np.NaN,
        'Srsoil': np.ones((n, duration))*np.NaN, 'Srres': np.ones((n, duration))*np.NaN, 'Srwp': np.ones((n, duration))*np.NaN, 'Vol0': np.ones((n))*np.NaN,
        'deltaRFa': np.ones((n, duration))
        }

    k = 0
    for t0 in tms:
        #print(t0)
        RF, RFr, S, Sr, Fco2, Frco2, Sattr = restoration_cases_new(ftype=ftype, rtype=rtype, duration=duration, start_yr=2025, tspan=t0)
        #RF, RFr, S, Sr, Fco2, Frco2, Sattr = restoration_cases_new(ftype='FNRnorth2', rtype='RME', duration=duration, start_yr=2025, tspan=t0)

        res4['t0'][k] = t0
        res4['Stree'][k,:] = S['Tree'].values
        res4['Ssoil'][k,:] = S['Soil'].values
        res4['Sres'][k,:] = S['Res']
        res4['Swp'][k,:] = S['WP']
        res4['Srsoil'][k,:] = Sr['Soil'].values
        res4['Srres'][k,:] = Sr['Res']
        res4['Srwp'][k,:] = Sr['WP']

        res4['Vol0'][k] = Sattr['Vol'].values[0]

        # site-level
        rfeco = RF['Soil'] + RF['Tree'] + RF['Res']
        rfreco = RFr['Soil'] +  RFr['Res']
        delta_rfeco = rfreco - rfeco

        #biome level, incl. wood use
        rfbio = RF['Soil'] + RF['Tree'] + RF['WP'] + RF['Res']
        rfrbio = RFr['Soil'] + RFr['WP']+ RFr['Res']
        delta_rfbio = rfrbio - rfbio

        if v == 'Tot':
            res4['deltaRFeco'][k,:] = delta_rfeco['Tot'].values.reshape((duration,))
            res4['deltaRFbio'][k,:] = delta_rfbio['Tot'].values.reshape((duration,)) + RF['deltaRFa'].reshape((duration,))
        else:
            res4['deltaRFeco'][k,:] = delta_rfeco[v].values.reshape((duration,))
            res4['deltaRFbio'][k,:] = delta_rfbio[v].values.reshape((duration,))
        k +=1

    # -----  plot figs ---------

    fig, ax = plt.subplots(1,2, figsize=(10,5))
    
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
        ax[0].plot(tt, aa, linestyle=ls, linewidth=lw, label='%.1f' %st)

    y0 = 0.0
    y1 = 4e-14
    y2 = -2e-14
    ax[0].fill_between(tt, y0, y1, color='r', alpha=0.2)
    ax[0].fill_between(tt, y2, y0, color='b', alpha=0.2)
    ax[0].set_ylim([y2, y1])
    ax[0].text(tt[4], 0.9*y1, 'a)')
    ax[0].set_xlim([min(tt), max(tt)+1])
    ax[0].set_ylabel('$\overline{\Delta RF}_{[0,t]}$ (W m$^{-2}$(earth) m$^{-2}$ (land restored))')
    ax[0].set_xlabel('years since restoration')
    #ax.plot(tt, np.zeros(len(tt)), 'k--', alpha=0.3)
    ax[0].legend(fontsize=10, frameon=False, title=r'$t_0/t_{rot}$')
    ax[0].set_title('Case 3: FRM --> Open mesotrophic.\n Restoration during stand rotation.')
    ax[0].text(tt[4], 0.9*y2, 'Restoration cooling', fontsize=8)
    ax[0].text(tt[4], 0.8*y1, 'Restoration warming', fontsize=8)

    tt = np.arange(0, duration, 1)

    for k in range(n4):
        if res4['t0'][k] < 22:
            ls = '-'
        else:
            ls = '--'
        st = res4['t0'][k]
        aa = np.cumsum(res4['deltaRFbio'][k,:]) / (tt +1)
        #aa = res4['deltaRFbio'][k,:]
        if k==0:
            lw = 2
        else:
            lw = 1.0
        ax[1].plot(tt, aa, linestyle=ls, linewidth=lw, label='%.2f' %st)


    ax[1].fill_between(tt, y0, y1, color='r', alpha=0.2)
    ax[1].fill_between(tt, y2, y0, color='b', alpha=0.2)
    ax[1].set_ylim([y2, y1])
    ax[1].text(tt[4], 0.9*y1, 'b)')
    ax[1].set_xlim([min(tt), max(tt)+1])
    ax[1].set_ylabel('$\overline{\Delta RF}_{[0,t]}$ (W m$^{-2}$(earth) m$^{-2}$ (land restored))')
    ax[1].set_xlabel('years since restoration')
    #ax.plot(tt, np.zeros(len(tt)), 'k--', alpha=0.3)
    ax[1].legend(fontsize=10, frameon=False, title=r'$\tau_{r}$ (yr)')
    ax[1].set_title('Case 4: FRM --> Open mesotrophic.\n Gradual change of F$_{soil}$.')
    ax[1].text(tt[4], 0.9*y2, 'Restoration cooling', fontsize=8)
    ax[1].text(tt[4], 0.8*y1, 'Restoration warming', fontsize=8)

    return fig

def make_fig4(duration=200):
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
    ax[0].set_ylabel('$\overline{\Delta RF}_{[0,t]}$ (W m$^{-2}$(earth) m$^{-2}$ (land restored))')
    ax[0].set_xlabel('years since restoration')
    
    #ax.plot(tt, np.zeros(len(tt)), 'k--', alpha=0.3)
    #ax[0].legend(fontsize=10, frameon=False, title=r'$t_0/t_{rot}$')
    ax[0].set_title('Nutrient rich forests')
    ax[0].text(tt[4], 0.9*y2, 'Restoration cooling', fontsize=8)
    ax[0].text(tt[4], 0.8*y1, 'Restoration warming', fontsize=8)

    ax[1].fill_between(tt, y0, y1, color='r', alpha=0.2)
    ax[1].fill_between(tt, y2, y0, color='b', alpha=0.2)
    ax[1].text(tt[4], 0.9*y1, 'b)')
    ax[1].set_ylim([y2, y1])
    ax[1].set_xlim([min(tt), max(tt)+1])
    ax[1].set_ylabel('$\overline{\Delta RF}_{[0,t]}$ (W m$^{-2}$(earth) m$^{-2}$ (land restored))')
    ax[1].set_xlabel('years since restoration')
    
    #ax.plot(tt, np.zeros(len(tt)), 'k--', alpha=0.3)
    #ax[0].legend(fontsize=10, frameon=False, title=r'$t_0/t_{rot}$')
    ax[1].set_title('Nutrient poor forests')
    ax[1].text(tt[4], 0.9*y2, 'Restoration cooling', fontsize=8)
    ax[1].text(tt[4], 0.8*y1, 'Restoration warming', fontsize=8)

    
    #--- Fertile peatland restoration pathways---

    rtypenames1 = ['Spruce mire', 'open eu/mesotrophic', 'open oligotrophic']
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

    rtypenames2 = ['Pine mire', 'open oligotrophic']#, 'open ombotrophic']
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
    
def make_fig_S4():
    # Soil net CO2 flux & albedo as function of stand volume

    from stand_carbon import soilCO2
    from radiative_forcing import albedo_delta_rf
    vol1 = np.arange(0,450, 1)
    vol2 = np.arange(0,450, 1)
    fnr_soilco2 = soilCO2(vol1, 'FNR')
    fnp_soilco2 = soilCO2(vol2, 'FNP')

    alb_rf = albedo_delta_rf(vol1)


    fig, ax = plt.subplots(1,2,figsize=(10,4.5))
    plt.subplots_adjust(wspace=0.25)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colo = prop_cycle.by_key()['color']
    colo = colo[0:6]

    tt = vol1
    tt = tt - min(tt)

    ax[0].plot(tt, np.zeros(len(tt)), 'k--', alpha=0.3)
    #ax2[0].plot(tt, 1e-3*stot, linewidth=2, label='Total')
    ax[0].plot(vol1, 44/12*fnr_soilco2, '-', color=colo[0], label='Nutrient rich')
    ax[0].plot(vol2, 44/12*fnp_soilco2, '-', color=colo[1], label='Nutrient poor')
    ax[0].legend(loc=4)
    ax[0].set_ylabel('F$_{co2, soil}$ (gCO2 m$^{-2}$ a$^{-1}$)')
    ax[0].set_xlabel('Vol (m$^{3}$ ha$^{-1}$)')
    #ax[0].set_ylim(-1, 500)
    #ax[0].set_xlim([min(tt), 81])
    ax[0].set_title('Soil CO$_2$ balance')

    ax[1].plot(vol1, alb_rf, '-', color=colo[2])
    #ax[1].legend(loc=2)
    ax[1].set_xlabel('Vol (m$^{3}$ ha$^{-1}$)')
    ax[1].set_ylabel('$\Delta$ RF$_{alb}$ (W m$^{-2}$(earth) m$^{-2}$ (land restored))')
    #ax[1].set_ylim(-1, 500)
    #ax[1].set_xlim([min(tt), 81])
    ax[1].set_title('albedo effect')

    ax[0].set_xlim([0, 450])
    ax[1].set_xlim([0, 450])

    return fig


def make_fig_S2():

    duration = 200
    tt = np.arange(0, duration, 1)

    fig, ax = plt.subplots(2,2,figsize=(12,10))
    plt.subplots_adjust(wspace=0.25)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colo = prop_cycle.by_key()['color']
    #colo = colo[0:7]

    # fertile peatlands
    ftypenames = ['Rhtk South', 'Mtkg South', 'Rhtkg North', 'Mtkg North']
    ftypes = ['FNRsouth1', 'FNRsouth2', 'FNRnorth1', 'FNRnorth2']
    #tmp1 = np.ones((n, duration))*np.NaN # biome scale
    k = 0
    for ft in ftypes:
        RF, RFr, S, Sr, Fco2, Frco2, Sattr = restoration_cases_new(ftype=ft, rtype='RME', duration=duration, start_yr=2025)

        if k >1:
            ls = '--'
        else:
            ls = '-'

        ax[0,0].plot(tt, Sattr['Vol'], linestyle=ls, color=colo[k], label=ftypenames[k])
        ax[0,0].set_ylabel('Vol (m$^{3}$ ha$^{-1}$)')
        ax[0,0].set_xlabel('Age (yr)')
        ax[0,0].set_ylim(-0.1, 500)
        ax[0,0].set_xlim([min(tt), 81])
        ax[0,0].set_title('Nutrient rich')

        ax[1,0].plot(tt, Sattr['BA'], linestyle=ls, color=colo[k], label=ftypenames[k])
        ax[1,0].set_ylabel('BA (m$^{2}$ ha$^{-1}$)')
        ax[1,0].set_xlabel('Age (yr)')
        ax[1,0].set_ylim(-0.1, 50)
        ax[1,0].set_xlim([min(tt), 81])
        #ax[1,0].set_title('Nutrient rich')

        k += 1

    ax[0,0].legend(fontsize=8, frameon=False, loc=2)
    ax[1,0].legend(fontsize=8, frameon=False, loc=2)

    # nutrient poor peatlands
    ftypenames = ['Ptkg South', 'Ptkg North']
    ftypes = ['FNPsouth', 'FNPnorth']
    #tmp1 = np.ones((n, duration))*np.NaN # biome scale
    k = 0
    for ft in ftypes:
        RF, RFr, S, Sr, Fco2, Frco2, Sattr = restoration_cases_new(ftype=ft, rtype='RME', duration=duration, start_yr=2025)

        if k == 1:
            ls = '--'
        else:
            ls = '-'
                
        ax[0,1].plot(tt, Sattr['Vol'], linestyle=ls, color=colo[k+4], label=ftypenames[k])
        ax[0,1].set_ylabel('Vol (m$^{3}$ ha$^{-1}$)')
        ax[0,1].set_xlabel('Age (yr)')
        ax[0,1].set_ylim(-0.1, 500)
        ax[0,1].set_xlim([min(tt), 81])
        ax[0,1].set_title('Nutrient poor')

        ax[1,1].plot(tt, Sattr['BA'], linestyle=ls, color=colo[k+4], label=ftypenames[k])
        ax[1,1].set_ylabel('BA (m$^{2}$ ha$^{-1}$)')
        ax[1,1].set_xlabel('Age (yr)')
        ax[1,1].set_ylim(-0.1, 50)
        ax[1,1].set_xlim([min(tt), 81])
        #ax[1,1].set_title('Nutrient poor')

        k += 1
    
    ax[0,1].legend(fontsize=8, frameon=False, loc=2)
    ax[1,1].legend(fontsize=8, frameon=False, loc=2)

    return fig
    #fig.savefig(r'Results/FNR_Mtkg_vol_ba.png', dpi=400)

def make_fig_S5():
    
    # example to break down radiative forcing calculation
    duration = 200
    tt = np.arange(0, duration, 1)
    fig, ax = plt.subplots(2,2,figsize=(12,10))

    plt.subplots_adjust(wspace=0.25)
    plt.subplots_adjust(hspace=0.25)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colo = prop_cycle.by_key()['color']
    #colo = colo[0:7]

    RF, RFr, S, Sr, Fco2, Frco2, Sattr  = restoration_cases_new(ftype='FNRsouth2', rtype='RME', duration=duration, start_yr=2025)

    #--- forest

    rf_bio = RF['Soil'] + RF['Tree'] + RF['Res'] + RF['WP']
    rf_eco = RF['Soil'] + RF['Tree'] + RF['Res']

    #-- restored peatland
    rrf_bio = RFr['Soil'] + RFr['Res'] + RFr['WP']
    rrf_eco = RFr['Soil'] + RFr['Res']

    yl = np.squeeze(np.minimum(rf_bio['Tot'].values, rrf_bio['Tot'].values))
    yh = np.squeeze(np.maximum(rf_bio['Tot'].values, rrf_bio['Tot'].values))

    ax[0,0].fill_between(tt, yl, yh, color=colo[0], alpha=0.3)
    #ax[0,0].plot(tt, np.zeros(len(tt)), 'k--')
    ax[0,0].plot(tt, rf_bio['Tot'].values, color=colo[0], linestyle='-', label='Nutrient rich forest')
    ax[0,0].plot(tt, rrf_bio['Tot'].values, color=colo[0], linestyle='--', label='Restored eu/mesotrophic')
    ax[0,0].set_ylim([0, 1.0e-13]) 

    yl = np.squeeze(np.minimum(rf_bio['CO2'].values, rrf_bio['CO2'].values))
    yh = np.squeeze(np.maximum(rf_bio['CO2'].values, rrf_bio['CO2'].values))
    
    ax[0,1].fill_between(tt, yl, yh, color=colo[2], alpha=0.3)
    #ax[0,1].plot(tt, np.zeros(len(tt)), 'k--')
    ax[0,1].plot(tt, rf_bio['CO2'].values, color=colo[2], linestyle='-', label='CO2')
    ax[0,1].plot(tt, rrf_bio['CO2'].values, color=colo[2], linestyle='--')
    ax[0,1].set_ylim([0, 1.0e-13]) 

    yl = np.squeeze(np.minimum(rf_bio['CH4'].values, rrf_bio['CH4'].values))
    yh = np.squeeze(np.maximum(rf_bio['CH4'].values, rrf_bio['CH4'].values))
    
    ax[1,0].fill_between(tt, yl, yh, color=colo[3], alpha=0.3)
    #ax[1,0].plot(tt, np.zeros(len(tt)), 'k--')
    ax[1,0].plot(tt, rf_bio['CH4'].values, color=colo[3], linestyle='-', label='CH4')
    ax[1,0].plot(tt, rrf_bio['CH4'].values, color=colo[3], linestyle='--')
    ax[1,0].set_ylim([0, 5e-14]) 

    yl = np.squeeze(np.minimum(rf_bio['N2O'].values, rrf_bio['N2O'].values))
    yh = np.squeeze(np.maximum(rf_bio['N2O'].values, rrf_bio['N2O'].values))
    
    ax[1,1].fill_between(tt, yl, yh, color=colo[1], alpha=0.3)
    #ax[1,1].plot(tt, np.zeros(len(tt)), 'k--')
    ax[1,1].plot(tt, rf_bio['N2O'].values, color=colo[1], linestyle='-', label='N2O')
    ax[1,1].plot(tt, rrf_bio['N2O'].values, color=colo[1], linestyle='--')
    ax[1,1].set_ylim([0, 1e-14])

    txt = ['Tot.', 'CO$_2$', 'CH$_4$', 'N$_2$O']
    m = 0
    for j in range(0,2):
        for k in range(0,2):
            ax[j,k].set_ylabel('RF (W m$^{-2}$(earth) m$^{-2}$ (land))')
            ax[j,k].set_xlabel('years')
            ax[j,k].set_title(txt[m])
            ax[j,k].set_xlim([-1, 201])

            m += 1
    ax[0,0].legend(fontsize=8, frameon=False)
    return fig

def make_fig2_ms(ftype1, rtype1, ftype2, rtype2, duration, ylims=[-1.e-13, 1.1e-13], savefig=False):

    growth_scenarios = {'FNRsouth1': ['South', 'Rhtkg', {'CO2': 265.0, 'CH4': 0.34, 'N2O': 0.23}],
                        'FNRsouth2': ['South', 'Mtkg', {'CO2': 265.0, 'CH4': 0.34, 'N2O': 0.23}],
                        'FNRnorth1': ['North', 'Rhtkg', {'CO2': 265.0, 'CH4': 0.34, 'N2O': 0.23}],
                        'FNRnorth2': ['North', 'Mtkg', {'CO2': 265.0, 'CH4': 0.34, 'N2O': 0.23}],
                        'FNPsouth': ['South', 'Ptkg', {'CO2': -45.0, 'CH4': 0.34, 'N2O': 0.08}],
                        'FNPnorth': ['North', 'Ptkg', {'CO2': -45.0, 'CH4': 0.34, 'N2O': 0.08}],
                    }
    
    v = growth_scenarios[ftype1]
    ss = ftype1[0:3]
    ofile = r'MS/Figs/Fig2_final.png'
    #ofile2 = r'MS/Figs/Case_1&2_%s_%s_attributes.png' %(ss, v[1] + ' ' + v[0])

    ttext1 = 'Case 1: FNR --> RME. Clear-cut at t=0.'
    #ttext0 = 'Case 1: %s (%s). Change in C storage since t=0.' %(v[1], v[0])

    v = growth_scenarios[ftype2]
    ss2 = ftype2[0:3]
    ttext2 = 'Case 2: FNR --> Spruce mire. No harvest at t=0.'

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
    ax[0,0].plot(tt, delta_rfbio['Tot'].values + RF1['deltaRFa'],'-', linewidth=2, label='Total')

    ax[0,0].plot(tt, delta_rfeco['Tot'].values + RF1['deltaRFa'], '--', linewidth=1.5, label='Total\n(no WP)')
    ax[0,0].plot(tt, delta_rfsoil['Tot'], '-', linewidth=1.5, label='Soil')

    ax[0,0].plot(tt, delta_rfL24['Tot'], 'k-',linewidth=1.0, alpha=0.5, label='Total\nL24')

    ax[0,0].legend(fontsize=8, frameon=False, loc=3)
    ax[0,0].set_ylabel('$\Delta RF$ (W m$^{-2}$(earth) m$^{-2}$ (land restored))')

    ax[0,0].set_xlim([min(tt), max(tt)+1])
    ax[0,0].set_ylim(ylims[0], ylims[1])

    #ax[0,0].text(tt[2], 0.9*y2, 'Restoration cooling', fontsize=8)
    #ax[0,0].text(tt[2], 0.9*y1, 'Restoration warming', fontsize=8)

    # contributions
    ax[0,1].fill_between(tt, y0, y1, color='r', alpha=0.2)
    ax[0,1].fill_between(tt, y2, y0, color='b', alpha=0.2)
    ax[0,1].text(tt[4], 0.9*ylims[1], 'b)')

    ax[0,1].set_title('Case 1: Contributions')
    #ax[0,1].plot(tt, np.zeros(len(tt)), 'k--', alpha=0.5)
    ax[0,1].plot(tt, delta_rfbio['Tot'].values + RF1['deltaRFa'],'-', linewidth=2, label='Total')
    ax[0,1].plot(tt, delta_rfbio['CO2'].values,'--', linewidth=1.5, label='CO$_2$')
    ax[0,1].plot(tt, delta_rfbio['CH4'].values, '--', linewidth=1.5, label='CH$_4$')
    ax[0,1].plot(tt, delta_rfbio['N2O'].values, '--', linewidth=1.5, label='N$_20$')
    ax[0,1].plot(tt, RF1['deltaRFa'], 'k-', linewidth=1.0, alpha=0.5, label='Albedo')
    #ax[0,1].plot(tt, delta_rfL24['Tot'], 'k-',linewidth=1.0, alpha=0.5, label='Total\nL24')

    #ax[0,1].plot(tt, RF1['deltaRFa'], '-', linewidth=1.0, label='Albedo')

    ax[0,1].legend(fontsize=8, frameon=False)
    ax[0,1].set_ylabel('$\Delta RF$ (W m$^{-2}$(earth) m$^{-2}$ (land restored))')
    #ax[0,1].set_xlabel('yr since restoration')
    ax[0,1].set_xlim([min(tt), max(tt)+1])
    ax[0,1].set_ylim(ylims[0], ylims[1])
    ax[0,1].text(tt[4], 0.85*ylims[0], 'Restoration cooling', fontsize=8)
    ax[0,1].text(tt[4], 0.8*ylims[1], 'Restoration warming', fontsize=8)

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

    ax[1,0].plot(tt, delta_rfbio2['Tot'].values + delta_rf_alb,'-', linewidth=2, label='Total')

    ax[1,0].plot(tt, delta_rfeco2['Tot'].values + delta_rf_alb, '--', linewidth=1.5, label='Total\n(no WP)')
    ax[1,0].plot(tt, delta_rfsoil2['Tot'], '-', linewidth=1.5, label='Soil')
    ax[1,0].plot(tt, delta_rfL242['Tot'], 'k-',linewidth=1.0, alpha=0.5, label='Total\nL24')
    #ax[0,0].plot(tt, RF2['deltaRFa'], '--', linewidth=1.5, label='Albedo')
    #ax[0,0].plot(tt, delta_rfL24['Tot'], 'k-',linewidth=1.0, alpha=0.5, label='L24')

    ax[1,0].legend(fontsize=8, frameon=False, loc=3)
    ax[1,0].set_ylabel('$\Delta RF$ (W m$^{-2}$(earth) m$^{-2}$ (land restored))')
    ax[1,0].set_xlabel('year since restoration')
    ax[1,0].set_xlim([min(tt), max(tt)+1])
    ax[1,0].set_ylim(ylims[0], ylims[1])
    #ax[0,0].text(tt[2], 0.9*y2, 'Restoration cooling', fontsize=8)
    #ax[0,0].text(tt[2], 0.9*y1, 'Restoration warming', fontsize=8)

    # contributions
    ax[1,1].fill_between(tt, y0, y1, color='r', alpha=0.2)
    ax[1,1].fill_between(tt, y2, y0, color='b', alpha=0.2)
    ax[1,1].text(tt[4], 0.9*ylims[1], 'd)')

    ax[1,1].set_title('Case 2: Contributions')
    #ax[0,1].plot(tt, np.zeros(len(tt)), 'k--', alpha=0.5)
    ax[1,1].plot(tt, delta_rfbio2['Tot'].values + delta_rf_alb,'-', linewidth=2, label='Total')
    ax[1,1].plot(tt, delta_rfbio2['CO2'].values,'--', linewidth=1.5, label='CO$_2$')
    ax[1,1].plot(tt, delta_rfbio2['CH4'].values, '--', linewidth=1.5, label='CH$_4$')
    ax[1,1].plot(tt, delta_rfbio2['N2O'].values, '--', linewidth=1.5, label='N$_20$')
    ax[1,1].plot(tt, delta_rf_alb, 'k-',linewidth=1.0, alpha=0.5, label='Albedo')

    ax[1,1].legend(fontsize=8, frameon=False)
    ax[1,1].set_ylabel('$\Delta RF$ (W m$^{-2}$(earth) m$^{-2}$ (land restored))')
    ax[1,1].set_xlabel('year since restoration')
    ax[1,1].set_xlim([min(tt), max(tt)+1])
    ax[1,1].set_ylim(ylims[0], ylims[1])
    ax[1,1].text(tt[4], 0.85*ylims[0], 'Restoration cooling', fontsize=8)
    ax[1,1].text(tt[4], 0.8*ylims[1], 'Restoration warming', fontsize=8)


    return fig

def timing_effect(ftype, rtype, t_rot, duration=200, v='Tot'):
    
    # Case 3: Does it matter when we restore? Restore FNR --> open RME at different parts of rotation cycle

    #duration = 200

    dt = np.fix(np.array([1.0, 0.8, 0.6, 0.4, 0.2])*t_rot)
    tms = 2025 + dt
    #print(tms)
    #tms = [t00, t00 + 10, t00 + 20, t00+30, t00+40, t00+50]
    
    n = len(tms)
    #select n first colors from colormap
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color'][0:n]


    res = {'t0': np.ones(n)*np.NaN, 'deltaRFbio': np.ones((n, duration))*np.NaN, 'deltaRFeco': np.ones((n, duration))*np.NaN, 'deltaRFsoil': np.ones((n, duration))*np.NaN,
        'Stree': np.ones((n, duration))*np.NaN, 'Ssoil': np.ones((n, duration))*np.NaN, 'Sres': np.ones((n, duration))*np.NaN, 'Swp': np.ones((n, duration))*np.NaN,
        'Srsoil': np.ones((n, duration))*np.NaN, 'Srres': np.ones((n, duration))*np.NaN, 'Srwp': np.ones((n, duration))*np.NaN, 't_rot': np.ones((n))*np.NaN,
        'deltaRFa': np.ones((n, duration)), 'Frco2': np.ones((n, duration)), 'Fco2': np.ones((n, duration), ),
        'RFbio': np.ones((n, duration)), 'RFeco': np.ones((n, duration)), 'rRFbio': np.ones((n, duration)), 'rRFeco': np.ones((n, duration)),
        }

    k = 0
    for t0 in tms:
        #print(t0)
        RF, RFr, S, Sr, Fco2, Frco2, Sattr = restoration_cases_new(ftype=ftype, rtype=rtype, duration=duration, start_yr=t0)

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
            res['RFeco'][k,:] = rfeco['Tot'].values.reshape((duration,))
            res['RFbio'][k,:] = rfbio['Tot'].values.reshape((duration,))
            
            res['rRFeco'][k,:] = rfreco['Tot'].values.reshape((duration,))
            res['rRFbio'][k,:] = rfrbio['Tot'].values.reshape((duration,))

            res['deltaRFeco'][k,:] = delta_rfeco['Tot'].values.reshape((duration,))
            res['deltaRFbio'][k,:] = delta_rfbio['Tot'].values.reshape((duration,)) + RF['deltaRFa'].reshape((duration,))
        else:
            res['RFeco'][k,:] = rfeco[v].values.reshape((duration,))
            res['RFbio'][k,:] = rfbio[v].values.reshape((duration,))
            res['rRFeco'][k,:] = rfreco[v].values.reshape((duration,))
            res['rRFbio'][k,:] = rfrbio[v].values.reshape((duration,))

            res['deltaRFeco'][k,:] = delta_rfeco[v].values.reshape((duration,))
            res['deltaRFbio'][k,:] = delta_rfbio[v].values.reshape((duration,))


        res['deltaRFa'][k,:] = RF['deltaRFa'].reshape((duration,))
        res['deltaRFsoil'][k,:] =  delta_rfsoil['CO2'].values.reshape((duration,))
        
        k +=1
    
    # -----  plot figs ---------

    fig, ax = plt.subplots(1,2, figsize=(10,5))

    tt = np.arange(0, duration, 1)

    for k in range(n):
        st = (res['t0'][k] - 2025) / t_rot
        if st <= 0.5:
            ls = '--'
        else:
            ls = '-'

        st = (res['t0'][k] - 2025) / t_rot

        dd = np.cumsum(res['deltaRFbio'][k,:]) / (tt +1)

        aa = res['deltaRFbio'][k,:]
        bb = res['RFbio'][k,:]
        cc = res['rRFbio'][k,:]

        print(cc)
        if k==0:
            lw = 2
        else:
            lw = 1.0

        ax[0].plot(tt, aa, color=colors[k], linestyle=ls, linewidth=lw, label='%.1f' %st)
        ax[1].plot(tt, bb, color=colors[k], linestyle='-', label='%.1f' %st)
        ax[1].plot(tt, cc, color=colors[k], linestyle='--')

    ax[0].set_title(v)
    ax[0].set_ylabel('$\Delta RF$ (W m$^{-2}$(earth) m$^{-2}$ (land restored))')
    ax[0].set_xlabel('years since restoration')
    ax[0].legend(fontsize=10, frameon=False, title=r'$t_0/t_{rot}$')

    ax[1].set_ylabel('$RF$ (W m$^{-2}$(earth) m$^{-2}$ (land ))')
    ax[1].set_xlabel('years since restoration')
    ax[1].legend(fontsize=10, frameon=False, title=r'$t_0/t_{rot}$')
    
    return fig

def make_fig_S3(duration, ylims=[-1.e-13, 1.1e-13], savefig=False):
    #compares restoration scenarios
    growth_scenarios = {'FNRsouth1': ['South', 'Rhtkg', {'CO2': 265.0, 'CH4': 0.34, 'N2O': 0.23}],
                        'FNRsouth2': ['South', 'Mtkg', {'CO2': 265.0, 'CH4': 0.34, 'N2O': 0.23}],
                        'FNRnorth1': ['North', 'Rhtkg', {'CO2': 265.0, 'CH4': 0.34, 'N2O': 0.23}],
                        'FNRnorth2': ['North', 'Mtkg', {'CO2': 265.0, 'CH4': 0.34, 'N2O': 0.23}],
                        'FNPsouth': ['South', 'Ptkg', {'CO2': -45.0, 'CH4': 0.34, 'N2O': 0.08}],
                        'FNPnorth': ['North', 'Ptkg', {'CO2': -45.0, 'CH4': 0.34, 'N2O': 0.08}],
                    }
    
    # Case 1: restore FNR --> open RME
    RF1, RFr1, S1, Sr1, Fco21, Frco21, Sattr1  = restoration_cases_new(ftype='FNRsouth1', rtype='RME', duration=duration, start_yr=2025)
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


    # Case 2: restoring without clear-cutting

    RF2, RFr2, S2, Sr2, Fco22, Frco22, Sattr2  = restoration_cases_new(ftype='FNRsouth1', rtype='RSM', duration=duration, start_yr=2025)
    # -- radiative forcings
    # albedo effect
    rfalb = RF1['deltaRFa'] # change in albedo RF in forestry relative to open peatland
    aa = max(Sattr2['Vol'].values)*np.ones(len(rfalb))
    rrfalb = albedo_delta_rf(aa) # change in RF of mature restored stand to open peatland
    delta_rf_alb = -1* (rrfalb - rfalb) # difference

    # site-level
    rfeco = RF2['Soil'] + RF2['Tree'] + RF2['Res']
    rfreco = RFr2['Soil']
    delta_rfeco2 = rfreco - rfeco + delta_rf_alb

    #biome level, incl. wood use        
    rfbio = RF2['Soil'] + RF2['Tree'] + RF2['WP'] + RF2['Res']
    rfrbio = RFr2['Soil']
    delta_rfbio2 = rfrbio - rfbio + delta_rf_alb

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

    ax[0,0].set_title('FNR (Rhtkg South) --> RME')
    ax[0,0].plot(tt, delta_rfbio['Tot'].values + RF1['deltaRFa'],'-', linewidth=2, label='Total')

    ax[0,0].plot(tt, delta_rfeco['Tot'].values + RF1['deltaRFa'], '--', linewidth=1.5, label='Total\n(no WP)')
    ax[0,0].plot(tt, delta_rfsoil['Tot'], '-', linewidth=1.5, label='Soil')

    ax[0,0].plot(tt, delta_rfL24['Tot'], 'k-',linewidth=1.0, alpha=0.5, label='Total\nL24')

    ax[0,0].legend(fontsize=8, frameon=False, loc=3)
    ax[0,0].set_ylabel('$\Delta RF$ (W m$^{-2}$(earth) m$^{-2}$ (land restored))')

    ax[0,0].set_xlim([min(tt), max(tt)+1])
    ax[0,0].set_ylim(ylims[0], ylims[1])

    # radiative forcings
    y0 = 0.0

    y1 = 1.1*np.max([np.abs(delta_rfsoil2['Tot'].values), np.abs(delta_rfbio2['Tot'].values), np.abs(delta_rfeco2['Tot'].values)])
    y2 = 1.1*np.min([delta_rfsoil2['Tot'].values, np.abs(delta_rfbio2['Tot'].values)])
    y1 = 1.5e-13
    y2 = -1.5e-13
    ax[1,0].fill_between(tt, y0, y1, color='r', alpha=0.2)
    ax[1,0].fill_between(tt, y2, y0, color='b', alpha=0.2)
    ax[1,0].text(tt[4], 0.9*ylims[1], 'c)')
    ax[1,0].set_title('FNR (Rhtkg South) --> Spruce mire')

    ax[1,0].plot(tt, delta_rfbio2['Tot'].values,'-', linewidth=2, label='Total')

    ax[1,0].plot(tt, delta_rfeco2['Tot'].values, '--', linewidth=1.5, label='Total\n(no WP)')
    ax[1,0].plot(tt, delta_rfsoil2['Tot'], '-', linewidth=1.5, label='Soil')
    ax[1,0].plot(tt, delta_rfL242['Tot'], 'k-',linewidth=1.0, alpha=0.5, label='Total\nL24')
    #ax[0,0].plot(tt, RF2['deltaRFa'], '--', linewidth=1.5, label='Albedo')
    #ax[0,0].plot(tt, delta_rfL24['Tot'], 'k-',linewidth=1.0, alpha=0.5, label='L24')

    #ax[1,0].legend(fontsize=8, frameon=False, loc=2)
    ax[1,0].set_ylabel('$\Delta RF$ (W m$^{-2}$(earth) m$^{-2}$ (land restored))')
    ax[1,0].set_xlabel('year since restoration')
    ax[1,0].set_xlim([min(tt), max(tt)+1])
    ax[1,0].set_ylim(ylims[0], ylims[1])
    
    #ax[0,0].text(tt[2], 0.9*ylims[0], 'Restoration cooling', fontsize=8)
    #ax[0,0].text(tt[2], 0.9*ylims[1], 'Restoration warming', fontsize=8)

    #ax[1,0].text(tt[2], 0.9*ylims[0], 'Restoration cooling', fontsize=8)
    #ax[1,0].text(tt[2], 0.9*ylims[1], 'Restoration warming', fontsize=8)

    # ----------- Mtkg North: low productivity -----------

        # Case 1: restore FNR --> open RME
    RF1, RFr1, S1, Sr1, Fco21, Frco21, Sattr1  = restoration_cases_new(ftype='FNRnorth2', rtype='RME', duration=duration, start_yr=2025)
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


    # Case 2: restoring without clear-cutting

    RF2, RFr2, S2, Sr2, Fco22, Frco22, Sattr2  = restoration_cases_new(ftype='FNRnorth2', rtype='RSM', duration=duration, start_yr=2025)
    # -- radiative forcings

    # albedo effect
    rfalb = RF1['deltaRFa'] # change in albedo RF in forestry relative to open peatland
    aa = max(Sattr2['Vol'].values)*np.ones(len(rfalb))
    rrfalb = albedo_delta_rf(aa) # change in RF of mature restored stand to open peatland
    delta_rf_alb = -1* (rrfalb - rfalb) # difference

    # site-level
    rfeco = RF2['Soil'] + RF2['Tree'] + RF2['Res']
    rfreco = RFr2['Soil']
    delta_rfeco2 = rfreco - rfeco + delta_rf_alb

    #biome level, incl. wood use        
    rfbio = RF2['Soil'] + RF2['Tree'] + RF2['WP'] + RF2['Res']
    rfrbio = RFr2['Soil']
    delta_rfbio2 = rfrbio - rfbio + delta_rf_alb

    # soil only
    rfsoil = RF2['Soil'] 
    rfrsoil = RFr2['Soil']
    delta_rfsoil2 = rfrsoil - rfsoil

    #laine 2024
    delta_rfL242 = RFr2['Soil'] - RF2['L24']

    ## --- plot figure

    rco2 = 44./12

    tt = S1['Tree'].index.values
    tt = tt - min(tt)

    # radiative forcings
    y0 = 0.0

    y1 = 1.1*np.max([np.abs(delta_rfsoil['Tot'].values), np.abs(delta_rfbio['Tot'].values), np.abs(delta_rfeco['Tot'].values)])
    y2 = 1.1*np.min([delta_rfsoil['Tot'].values, np.abs(delta_rfbio['Tot'].values)])
    y1 = 1.5e-13
    y2 = -1.5e-13
    ax[0,1].fill_between(tt, y0, y1, color='r', alpha=0.2)
    ax[0,1].fill_between(tt, y2, y0, color='b', alpha=0.2)
    ax[0,1].text(tt[4], 0.9*ylims[1], 'b)')

    ax[0,1].set_title('FNR (Mtkg North) --> RME')
    ax[0,1].plot(tt, delta_rfbio['Tot'].values + RF1['deltaRFa'],'-', linewidth=2, label='Total')

    ax[0,1].plot(tt, delta_rfeco['Tot'].values + RF1['deltaRFa'], '--', linewidth=1.5, label='Total\n(no WP)')
    ax[0,1].plot(tt, delta_rfsoil['Tot'], '-', linewidth=1.5, label='Soil')

    ax[0,1].plot(tt, delta_rfL24['Tot'], 'k-',linewidth=1.0, alpha=0.5, label='Total\nL24')

    ax[0,1].legend(fontsize=8, frameon=False, loc=3)
    ax[0,1].set_ylabel('$\Delta RF$ (W m$^{-2}$(earth) m$^{-2}$ (land restored))')

    ax[0,1].set_xlim([min(tt), max(tt)+1])
    ax[0,1].set_ylim(ylims[0], ylims[1])

    # radiative forcings
    y0 = 0.0

    y1 = 1.1*np.max([np.abs(delta_rfsoil2['Tot'].values), np.abs(delta_rfbio2['Tot'].values), np.abs(delta_rfeco2['Tot'].values)])
    y2 = 1.1*np.min([delta_rfsoil2['Tot'].values, np.abs(delta_rfbio2['Tot'].values)])
    y1 = 1.5e-13
    y2 = -1.5e-13
    ax[1,1].fill_between(tt, y0, y1, color='r', alpha=0.2)
    ax[1,1].fill_between(tt, y2, y0, color='b', alpha=0.2)
    ax[1,1].text(tt[4], 0.9*ylims[1], 'd)')
    ax[1,1].set_title('FNR (Mtkg North) --> Spruce mire')

    ax[1,1].plot(tt, delta_rfbio2['Tot'].values,'-', linewidth=2, label='Total')

    ax[1,1].plot(tt, delta_rfeco2['Tot'].values, '--', linewidth=1.5, label='Total\n(no WP)')
    ax[1,1].plot(tt, delta_rfsoil2['Tot'], '-', linewidth=1.5, label='Soil')
    ax[1,1].plot(tt, delta_rfL242['Tot'], 'k-',linewidth=1.0, alpha=0.5, label='Total\nL24')
    #ax[0,0].plot(tt, RF2['deltaRFa'], '--', linewidth=1.5, label='Albedo')
    #ax[0,0].plot(tt, delta_rfL24['Tot'], 'k-',linewidth=1.0, alpha=0.5, label='L24')

    #ax[1,1].legend(fontsize=8, frameon=False, loc=2)
    ax[1,1].set_ylabel('$\Delta RF$ (W m$^{-2}$(earth) m$^{-2}$ (land restored))')
    ax[1,1].set_xlabel('year since restoration')
    ax[1,1].set_xlim([min(tt), max(tt)+1])
    ax[1,1].set_ylim(ylims[0], ylims[1])
    
    ax[0,1].text(tt[-60], 0.9*ylims[0], 'Restoration cooling', fontsize=8)
    ax[0,1].text(tt[-60], 0.9*ylims[1], 'Restoration warming', fontsize=8)

    ax[1,1].text(tt[-60], 0.9*ylims[0], 'Restoration cooling', fontsize=8)
    ax[1,1].text(tt[-60], 0.9*ylims[1], 'Restoration warming', fontsize=8)

    return fig

### ***************************

def make_fig4(duration=200):
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
    ax[0].set_ylabel('$\overline{\Delta RF}_{[0,t]}$ (W m$^{-2}$(earth) m$^{-2}$ (land restored))')
    ax[0].set_xlabel('years since restoration')
    
    #ax.plot(tt, np.zeros(len(tt)), 'k--', alpha=0.3)
    #ax[0].legend(fontsize=10, frameon=False, title=r'$t_0/t_{rot}$')
    ax[0].set_title('Nutrient rich forests')
    ax[0].text(tt[4], 0.9*y2, 'Restoration cooling', fontsize=8)
    ax[0].text(tt[4], 0.8*y1, 'Restoration warming', fontsize=8)

    ax[1].fill_between(tt, y0, y1, color='r', alpha=0.2)
    ax[1].fill_between(tt, y2, y0, color='b', alpha=0.2)
    ax[1].text(tt[4], 0.9*y1, 'b)')
    ax[1].set_ylim([y2, y1])
    ax[1].set_xlim([min(tt), max(tt)+1])
    ax[1].set_ylabel('$\overline{\Delta RF}_{[0,t]}$ (W m$^{-2}$(earth) m$^{-2}$ (land restored))')
    ax[1].set_xlabel('years since restoration')
    
    #ax.plot(tt, np.zeros(len(tt)), 'k--', alpha=0.3)
    #ax[0].legend(fontsize=10, frameon=False, title=r'$t_0/t_{rot}$')
    ax[1].set_title('Nutrient poor forests')
    ax[1].text(tt[4], 0.9*y2, 'Restoration cooling', fontsize=8)
    ax[1].text(tt[4], 0.8*y1, 'Restoration warming', fontsize=8)

    
    #--- Fertile peatland restoration pathways---

    rtypenames1 = ['Spruce mire', 'open eu/mesotrophic', 'open oligotrophic']
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

    rtypenames2 = ['Pine mire', 'open oligotrophic']#, 'open ombotrophic']
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

def make_sa_fig(duration=200):

    rco2 = 1000 * 44./12. # from kg C m-2 to g CO2 m-2
    cf = 1e-12 #1 g gas as Mton gas

    
    # soil net GHG balances (Laine et al. 2024 Table 1). Unit = g gas m-2 a-1
    Fsoil = {'FNR': {'CO2': 265.0, 'CH4': 0.34, 'N2O': 0.23}, # forest, nutrient rich
            'FNP': {'CO2': -45.0, 'CH4': 0.34, 'N2O': 0.08}, # forest, nutrient poor
            'RSM': {'CO2': -91.0, 'CH4': 1.70, 'N2O': 0.10}, # restored, spruce mire
            'RPM': {'CO2': -97.0, 'CH4': 4.80, 'N2O': 0.03}, # restored, pine mire
            'RME': {'CO2': -104.0, 'CH4': 15.00, 'N2O': 0.10}, # restored, open eu/mesotrophic
            'ROL': {'CO2': -124.0, 'CH4': 22.00, 'N2O': 0.03}, # restored, open oligotrophic
            'ROM': {'CO2': -95.0, 'CH4': 9.70, 'N2O': 0.03}, # restored, open ombotrophics
            }
    
    

    # Case 1: restore FNR --> open RME: with default restored peatland soil CO2
    RF1, RFr1, S1, Sr1, Fco21, Frco21, Sattr1  = restoration_cases_new(ftype='FNRsouth2', rtype='RME', duration=duration, start_yr=2025)
    
    RF2, RFr2, _, _, _, _, _  = restoration_cases_new(ftype='FNRsouth1', rtype='RME', duration=duration, start_yr=2025)
    
    RF3, RFr3, _, _, _, _, _  = restoration_cases_new(ftype='FNRnorth2', rtype='RME', duration=duration, start_yr=2025)

    tvec = RF1['Tree'].index

    # compute restored peatland RF's with altered CO2, CH4 and N2O budgets
    
    Fc0 = -104.0
    Fm0 = 15.0
    Fn0 = 0.10
    
    fc = [1.0, 0.5, -0.5, -1.0]
    fm = [0.5, 0.25, -0.25, -0.5]
    
    RFrc = [None, None, None, None]
    j = 0
    for k in fc:    
        fghg = pd.DataFrame(data=np.zeros((duration, 3)), columns=['CO2', 'CH4', 'N2O'], index=tvec)
        fghg['CO2'] = Fc0*(1 + k) * cf
        fghg['CH4'] = Fm0 * cf #Fm0*(1 + k) * cf
        fghg['N2O'] = Fn0 * cf
    
        aa = radiative_forcing(fghg)
        RFrc[j] = aa
        j += 1

    RFrm = [None, None, None, None]
    j = 0
    for k in fm:    
        fghg = pd.DataFrame(data=np.zeros((duration, 3)), columns=['CO2', 'CH4', 'N2O'], index=tvec)
        fghg['CO2'] = Fc0  * cf
        fghg['CH4'] = Fm0*(1 + k) * cf
        fghg['N2O'] = Fn0 * cf
        
        aa = radiative_forcing(fghg)
        RFrm[j] = aa
        j += 1
    # -- radiative forcings
    #print(len(RFrm), RFrm[0], aa)


    #biome level, incl. wood use
    rfbio = RF1['Soil'] + RF1['Tree'] + RF1['WP'] + RF1['Res']
    rfrbio = RFr1['Soil']['Tot'] + RFr1['WP']+ RFr1['Res']

    yy = rfrbio['Tot'] - rfbio['Tot']
    
    ## FIGURE

    fig, ax = plt.subplots(2,1, figsize=(8,10))
    tt = np.arange(0, duration, 1)
    
    #select 6 first colors from colormap
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color'][0:6]
    
    #CH4

    rfmbio = RFrm[0]['Tot'] + RFr1['WP']+ RFr1['Res']
    y0 = np.squeeze((rfmbio['Tot'] - rfbio['Tot'] ).values)

    rfmbio = RFrm[-1]['Tot'] + RFr1['WP']+ RFr1['Res']
    y1 = np.squeeze((rfmbio['Tot'] - rfbio['Tot'] ).values)
    ax[0].fill_between(tt, y0, y1, color='r', alpha=0.2)

    rfmbio = RFrm[1]['Tot'] + RFr1['WP']+ RFr1['Res']
    y0 = np.squeeze((rfmbio['Tot'] - rfbio['Tot'] ).values)

    rfmbio = RFrm[2]['Tot'] + RFr1['WP']+ RFr1['Res']
    y1 = np.squeeze((rfmbio['Tot'] - rfbio['Tot'] ).values)
    ax[0].fill_between(tt, y0, y1, color='r', alpha=0.4)

    # CO2
    rfrbio = RFrc[0]['Tot'] + RFr1['WP']+ RFr1['Res']
    y0 = np.squeeze((rfrbio['Tot']  - rfbio['Tot']).values)

    rfrbio = RFrc[3]['Tot'] + RFr1['WP']+ RFr1['Res']
    y1 = np.squeeze((rfrbio['Tot'] - rfbio['Tot'] ).values)
    print(np.shape(RFrc))
    ax[0].fill_between(tt, y0, y1, color='g', alpha=0.2)

    rfrbio = RFrc[1]['Tot'] + RFr1['WP']+ RFr1['Res']
    y0 = np.squeeze((rfrbio['Tot']  - rfbio['Tot']).values)

    rfrbio = RFrc[2]['Tot'] + RFr1['WP']+ RFr1['Res']
    y1 = np.squeeze((rfrbio['Tot'] - rfbio['Tot'] ).values)
    ax[0].fill_between(tt, y0, y1, color='g', alpha=0.3)
    
    ax[0].plot(tt, yy, 'k-', linewidth=2)
    
    ax[0].plot([-1, 201], [0, 0], 'k--', linewidth=1, alpha=0.5)
    
    ax[0].set_xlim([0, 200])
    ax[0].set_ylim([-6e-14, 6e-14])
    ax[0].set_ylabel('$\Delta RF$ (W m$^{-2}$(earth) m$^{-2}$ (land restored))')
    ax[0].set_xlabel('years since restoration')

    ax[0].text(tt[4], 0.9*-6e-14, 'Restoration cooling', fontsize=8)
    ax[0].text(tt[4], 0.9*6e-14, 'Restoration warming', fontsize=8)

    # ax[1]
    rfbio = RF1['Soil'] + RF1['Tree'] + RF1['WP'] + RF1['Res']
    rfrbio = RFr1['Soil']['Tot'] + RFr1['WP']+ RFr1['Res']

    yy = rfrbio['Tot'] - rfbio['Tot']

    tmp = np.zeros((duration, 3)) + np.NaN

    tmp[:,0] = np.squeeze(yy.values)
    
    rfbio = RF2['Soil'] + RF2['Tree'] + RF2['WP'] + RF2['Res']
    rfrbio = RFr2['Soil']['Tot'] + RFr2['WP']+ RFr2['Res']
    tmp[:,1] = np.squeeze((rfrbio['Tot'] - rfbio['Tot']).values)

    rfbio = RF3['Soil'] + RF3['Tree'] + RF3['WP'] + RF3['Res']
    rfrbio = RFr3['Soil']['Tot'] + RFr3['WP']+ RFr3['Res']
    tmp[:,2] = np.squeeze((rfrbio['Tot'] - rfbio['Tot']).values)

    y1_l = np.min(tmp, axis=1)
    y1_h = np.max(tmp, axis=1)
    
    ax[1].fill_between(tt, y1_l, y1_h, color='b', alpha=0.3)
    ax[1].plot(tt, tmp[:,1], 'b-', linewidth=1.5, alpha=0.3)
    ax[1].plot(tt, tmp[:,2], 'b--', linewidth=1.5, alpha=0.5)
    ax[1].plot(tt, yy, 'k-', linewidth=2)
    
    ax[1].plot([-1, 201], [0, 0], 'k--', linewidth=1, alpha=0.5)
    
    ax[1].text(tt[4], 0.9*-6e-14, 'Restoration cooling', fontsize=8)
    ax[1].text(tt[4], 0.9*6e-14, 'Restoration warming', fontsize=8)

    ax[1].set_xlim([0, 200])
    ax[1].set_ylim([-6e-14, 6e-14])
    ax[1].set_ylabel('$\Delta RF$ (W m$^{-2}$(earth) m$^{-2}$ (land restored))')
    ax[1].set_xlabel('years since restoration')

    return fig, ax
