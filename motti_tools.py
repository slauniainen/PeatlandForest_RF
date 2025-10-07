# massage motti-outputs for Radiative Forcing calculations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ffile = r'Data/Ruotsinkylä_Mtkg.xlsx'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from io import StringIO
from itertools import takewhile

EPS = 1e-6

# conversion from ton DM ha-1 to g C m-2
cf = 1e6 / 1e4 * 0.5
c_to_co2 = 44./12

v_units = {'simulointi': '-', 'vuosi': '-',	'Ika': 'a', 'N': 'ha-1',
	'keskiTilavuus': '?', 'PPA': 'm2 ha-1',	
	'Hg': 'm', 'Dg': 'cm', 'Ha': 'm', 'Da': 'm',
	'hdom': 'm', 'tilavuus': 'm3 ha-1',	'puustonArvo': '€',	
	'tukki': 'm3 ha-1', 'kuitu': 'm3 ha-1', 'hukka': 'm3 ha-1',
	'runko aines': 'ton ha-1', 'runko hukka': 'ton ha-1', 'elävät oksat': 'ton ha-1', 'kuolleet oksat': 'ton ha-1',
	'lehdet': 'ton ha-1', 'kannot': 'ton ha-1', 'juuret_karkea': 'ton ha-1', 'juuret_hieno': 'ton ha-1', 'biom_yht': 'ton ha-1',
	'hiilivarasto':	'ton CO2 ha-1',
	'Mustikkasato': 'unknown', 'Puolukkasato': 'unknown', 'Mustikkapeitto': 'unknown', 'Puolukkapeitto': 'unknown', 
	'Kuolleisuus': 'm3 ha-1', 'Tuotos': 'm3 ha-1', 'KuollutBiomassa': 'm3 ha-1', 'TuotosHiilenä': 'ton CO2 ha-1',
	'N_ma': 'ha-1',	'PPA_ma': 'm2 ha-1', 'Hg_ma': 'm', 'Dg_ma': 'cm', 'HDom_ma': 'm', 'tilavuus_ma': 'm3ha-1', 'tukki_ma': 'm3ha-1', 'kuitu_ma': 'm3ha-1',	
	'N_ku': 'ha-1',	'PPA_ku': 'm2 ha-1', 'Hg_ku': 'm', 'Dg_ku': 'cm', 'HDom_ku': 'm', 'tilavuus_ku': 'm3 ha-1', 'tukki_ku': 'm3 ha-1', 'kuitu_ku': 'm3 ha-1',
	'N_ra': 'ha-1',	'PPA_ra': 'm2 ha-1', 'Hg_ra': 'm', 'Dg_ra': 'cm', 'HDom_ra': 'm', 'tilavuus_ra': 'm3 ha-1', 'tukki_ra': 'm3 ha-1', 'kuitu_ra': 'm3 ha-1',
	'N_hi': 'ha-1',	'PPA_hi': 'm2 ha-1', 'Hg_hi': 'm', 'Dg_hi': 'cm', 'HDom_hi': 'm', 'tilavuus_hi': 'm3 ha-1', 'tukki_hi': 'm3 ha-1', 'kuitu_hi': 'm3 ha-1',
	'laho_tilavuus': 'm3 ha-1', 'laho_kantomassa':	'unknown', 'laho_juurimassa': 'unknown', 'laho_oksamassa': 'unknown', 'laho_runkomassa': 'unknown',	
	'laho_massa_yht': 'unknown', 'laho_hiili_yht': 'ton CO2 ha-1', 'puusto_hiili_yht': 'ton CO2 ha-1', 'diversiteetti': '-'
	}

usecols= {'Ika': 'a', 'N': 'ha-1', 'PPA': 'm2 ha-1',	
	'Hg': 'm', 'Dg': 'cm', 'hdom': 'm', 'tilavuus': 'm3 ha-1', 'puustonArvo': '€',	
	'tukki': 'm3 ha-1', 'kuitu': 'm3 ha-1', 'hukka': 'm3 ha-1',
	'runko aines': 'g C m-2 a-1', 'runko hukka': 'g C m-2 a-1', 'elävät oksat': 'g C m-2 a-1', 'kuolleet oksat': 'g C m-2 a-1',
	'lehdet': 'g C m-2 a-1', 'kannot': 'g C m-2 a-1', 'juuret_karkea': 'g C m-2 a-1', 'juuret_hieno': 'g C m-2 a-1', 'biom_yht': 'g C m-2 a-1',
	'hiilivarasto':	'g C m-2 a-1', 'puusto_hiili_yht': 'g C m-2 a-1', 'laho_hiili_yht':	'g C m-2 a-1'
	}

subset_cols= ['Ika', 'PPA', 'Dg', 'hdom', 'tilavuus', 'puustonArvo', 'FFol', 'FWD', 'CWD', 'NPP']


#%% Interpolate all columns in a dataframe to denser index

def interp_to_denser_index(df, new_index):
    """Return a new DataFrame with all columns values linearly interpolated
    to the new_index values."""
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for colname, col in df.iteritems():
        #print(colname, col)
        df_out[colname] = np.interp(new_index, df.index, col)

    return df_out


def read_csv_until_column_change(filepath, delimiter=";"):
    with open(filepath, 'r') as f:
        # Read the first line to determine the expected number of columns
        first_line = f.readline()
        expected_cols = len(first_line.strip().split(delimiter))

        # Use takewhile to read lines with matching column count
        valid_lines = [first_line] + list(takewhile(
            lambda line: len(line.strip().split(delimiter)) == expected_cols,
            f
        ))

    # Load into DataFrame
    dat = pd.read_csv(StringIO(''.join(valid_lines)), delimiter=delimiter)
    #print(dat.head())
    #print(dat.tail())

    return dat


#%% Read and massage 'LukeMotti' outputs to create annual datafile for RF calculations

def massage_LukeMotti(ffile, CCF=False, timespan=300):

    #ffile = r'Data/Ruotsinkylä_Mtkg.xlsx'

    # read file
    raw = pd.read_excel(ffile, sheet_name='Metsikkötaulu')

    # remove duplicate rows
    raw.drop_duplicates(inplace=True)

    # convert biomasses ton ha-1 --> g C m-2
    for c in ['runko aines', 'runko hukka', 'elävät oksat', 'kuolleet oksat', 'lehdet', 'kannot', 'juuret_karkea', 'juuret_hieno', 'biom_yht']:
        raw[c] = raw[c] * cf

    # hiilivarasto ton CO2 ha-1 --> g C m-2
    raw['hiilivarasto'] = raw['hiilivarasto'] * 100 / c_to_co2
    raw['puusto_hiili_yht'] = raw['puusto_hiili_yht'] * 100 / c_to_co2
    raw['laho_hiili_yht'] = raw['laho_hiili_yht'] * 100 / c_to_co2

    # in years when stand has been harvested, there are values pre- and post-harvest. Add +1 to post-harvest year
    for k in range(1, len(raw)):
        if raw['vuosi'].iloc[k] == raw['vuosi'].iloc[k-1]:
            raw['vuosi'].iloc[k] += 1
            raw['Ika'].iloc[k] += 1

    raw.index = raw['vuosi']
    #raw.reset_index(inplace=True, drop=True)

    # save initial state of the stand
    initial_state = raw.iloc[0].copy()


    # if simulations start from bare ground, tilavuus is zero until large trees emerge. In this case, interpolate tilavuus and biom_yht based on 'PPA'
    if CCF == False:
        vol = raw['tilavuus'].values
        ba = raw['PPA'].values
        bm = raw['biom_yht'].values

        ix = np.min(np.nonzero(vol[1:])) + 1
        #print(ix)
        if ix > 0:
            aa = vol[ix] / ba[ix]
            #print(aa)
            vol[1:ix] = aa * ba[1:ix]
            raw['tilavuus'] = vol

            bb = bm[ix] / ba[ix]
            #print(aa)
            bm[1:ix] = bb * ba[1:ix]
            raw['biom_yht'] = bm
            del ix, aa#, bb

        del vol, ba, bm
    

    # interpolate linearly to annual values
    tmp = np.arange(0, max(raw['vuosi']+1), 1)
    
    # annual motti-file
    dat = interp_to_denser_index(raw, tmp)
    dat['Ika'] = np.floor(dat['Ika'].values)
    dat = dat[usecols]

    #dat.iloc[0] = initial_state[usecols]
    
    # --- calculate harvest removals ---
    dat = dat.reindex(columns=dat.columns.tolist() + ['poistuma_vol', 'poistuma_runko aines', 'poistuma_runko hukka', 
                                                    'poistuma_elävät oksat', 'poistuma_kuolleet oksat', 
                                                    'poistuma_lehdet', 'poistuma_kannot', 'poistuma_juuret_karkea', 
                                                    'poistuma_juuret_hieno', 'poistuma_biom_yht', 'poistuma_tukki', 'poistuma_kuitu', 
                                                    'NPP', 'FFol', 'FWD', 'CWD'])

    dat[['poistuma_vol', 'poistuma_runko aines', 'poistuma_runko hukka', 'poistuma_elävät oksat', 'poistuma_kuolleet oksat', 
         'poistuma_lehdet', 'poistuma_kannot', 'poistuma_juuret_karkea', 'poistuma_juuret_hieno', 'poistuma_biom_yht', 'poistuma_tukki', 
         'poistuma_kuitu', 'NPP', 'FFol', 'FWD', 'CWD']] = 0.0
    
    for k in range(1, len(dat)):
        dV = dat['tilavuus'].iloc[k] - dat['tilavuus'].iloc[k-1]
        if dV < 0:
            dat['poistuma_vol'].iloc[k-1] = -dV
            for c in ['runko aines', 'runko hukka', 'elävät oksat', 'kuolleet oksat', 'lehdet', 'kannot', 'juuret_karkea', 'juuret_hieno', 'biom_yht']:
                dX = dat[c].iloc[k] - dat[c].iloc[k-1]
                dat['poistuma_' + c].iloc[k-1] = -dX

            # divide poistuma_runko aines between tukki and kuitu based on tukki & kuitu volume's
            sf = dat['tukki'].iloc[k-1] / (dat['tukki'].iloc[k-1] + dat['kuitu'].iloc[k-1] + EPS)
            dat['poistuma_tukki'] = dat['poistuma_runko aines'] * sf
            dat['poistuma_kuitu'] = dat['poistuma_runko aines'] * (1 - sf)

    # biomass growth (NPP, g C m-2 a-1)
    dat['NPP'] = 0.0
    for k in range(1, len(dat)):
        dX = dat['biom_yht'].iloc[k] - dat['biom_yht'].iloc[k-1]
        #dX = dat['hiilivarasto'].iloc[k] - dat['hiilivarasto'].iloc[k-1]
        if dX > 0:
            dat['NPP'].iloc[k] = dX

    dat['NPP'].interpolate('linear', inplace=True)
    
    #ix = np.where(np.isnan(dat['NPP']))[0]
    #if len(ix) > 0:
    #    dat['NPP'].iloc[ix] = 0.0

    # litter inputs from harvests: foliage, FWD, CWD: this follows Ambio-paper

    dat['FFol'] = dat['poistuma_lehdet'] + dat['poistuma_juuret_hieno'] 
    dat['FWD'] = dat['poistuma_elävät oksat'] + dat['poistuma_kuolleet oksat'] + dat['poistuma_juuret_karkea']
    dat['CWD'] = dat['poistuma_kannot'] + dat['poistuma_runko hukka']

    ix = np.where(np.isnan(dat['FFol']) == True)[0]
    if len(ix) > 0:
        dat['FFol'].iloc[ix] = 0.0
        dat['FWD'].iloc[ix] = 0.0
        dat['CWD'].iloc[ix] = 0.0 

    data = dat.copy()

    if CCF == False:
        # -- rotation length---
        rl = np.max(dat['Ika'])
        #dat['Ika'].iloc[-1] = 0
        #cycle for multiple rotations
        tmp = dat.iloc[1:-1]
        
        rl = np.max(data['Ika'])
        cycles = np.int(np.ceil(timespan/rl)) -1
        #print(cycles)
        for n in range(cycles):
            data = pd.concat([data, tmp])

    data.reset_index(inplace=True)
    data = data.iloc[0:timespan]

    # rename columns to match those of OptiMotti-outputs

    out = pd.DataFrame(data = None, index=data.index, 
                       columns=['year', 'DomAge', 'DomHeight', 'Hg', 'BA', 'N', 'Dg', 'vol', 'sawVol',
                                'pulpVol', 'bioTot', 'bioStemComm', 'bioStemWaste', 'bioBranches',
                                'bioFoliage', 'bioStumps', 'bioRootsC', 'bioRootsF', 'harvest_vol',
                                'harvest_bioFoliage', 'harvest_bioRootsF', 'harvest_bioRootsC',
                                'harvest_bioBranches', 'harvest_bioStumps', 'harvest_bioTot',
                                'harvest_bioStemComm', 'harvest_bioStemWaste', 'harvest_Log',
                                'harvest_Fibre', 'NPP', 'FFol', 'FWD', 'CWD']
                        )
    out['year'] = data['vuosi']; out['DomAge'] = data['Ika']; out['DomHeight'] = data['hdom']
    out[['N', 'Hg', 'Dg']] = data[['N', 'Hg', 'Dg']]
    out['BA'] = data['PPA']; out['vol'] = data['tilavuus']
    
    out['sawVol'] = data['tukki']; out['pulpVol'] = data['kuitu'] 
    
    c  =  ['bioTot', 'bioStemComm', 'bioStemWaste', 'bioFoliage', 'bioStumps', 'bioRootsC', 'bioRootsF']
    out[c] = data[['biom_yht', 'runko aines', 'runko hukka', 'lehdet', 'kannot', 'juuret_karkea', 'juuret_hieno']]
    
    out['bioBranches'] = data['elävät oksat'] + data['kuolleet oksat']

    d = ['harvest_vol', 'harvest_bioFoliage', 'harvest_bioRootsF', 'harvest_bioRootsC',
        'harvest_bioStumps', 'harvest_bioTot',
          'harvest_bioStemComm', 'harvest_bioStemWaste', 'harvest_Log',
           'harvest_Fibre', 'NPP', 'FFol', 'FWD', 'CWD']
    
    out[d] = data[['poistuma_vol', 'poistuma_lehdet', 'poistuma_juuret_hieno', 'poistuma_juuret_karkea', 
                   'poistuma_kannot', 'poistuma_biom_yht',
                   'poistuma_runko aines', 'poistuma_runko hukka', 'poistuma_tukki', 
                   'poistuma_kuitu', 'NPP', 'FFol', 'FWD', 'CWD']]
    
    out['harvest_bioBranches'] = data['poistuma_elävät oksat'] + data['poistuma_kuolleet oksat']
    
    out['year'] = out.index.values
    
    return out


def massage_OptiMotti(ffile, CCF=False, timespan=300):

    cols = ['year', 'period', 'DomAge', 'DomHeight', 'Hg', 'BA', 'N', 'Dg', 'mainSpecies', 'vol', 'sawVol', 'pulpVol', 'biomass', 
            'bioTotUt', 'bioStemCommUt', 'bioStemWasteUt', 'bioBranchesLUt', 'bioBranchesDUt', 'bioFoliageUt', 'bioStumpsUt', 'bioRootsCUt', 'bioRootsFUt', 
            'bioTot12', 'bioStemComm12', 'bioStemWaste12', 'bioBranchesL12', 'bioBranchesD12', 'bioFoliage12', 'bioStumps12', 'bioRootsC12', 'bioRootsF12', 
            'bioTot3', 'bioStemComm3', 'bioStemWaste3', 'bioBranchesL3', 'bioBranchesD3', 'bioFoliage3', 'bioStumps3', 'bioRootsC3', 'bioRootsF3', 'bioTot4', 
            'bioStemComm4', 'bioStemWaste4', 'bioBranchesL4', 'bioBranchesD4', 'bioFoliage4', 'bioStumps4', 'bioRootsC4', 'bioRootsF4']


    usecols = ['year', 'DomAge', 'DomHeight', 'Hg', 'BA', 'N', 'Dg', 'vol', 'sawVol', 'pulpVol', 
               'bioTot', 'bioStemComm', 'bioStemWaste', 'bioBranches', 'bioFoliage', 'bioStumps', 'bioRootsC', 'bioRootsF']

    # read file
    #ffile = r'Data/Development_opt_tst.csv'
    
    raw = read_csv_until_column_change(ffile, delimiter=';')
    #raw = pd.read_csv(ffile, sep=',')
    ix = np.where(np.isfinite(raw['year']))[0]
    raw = raw.iloc[ix]


    # remove duplicate rows
    raw.drop_duplicates(inplace=True)
    raw.index = raw['year']

    # in years when stand has been harvested, there are values pre- and post-harvest. Add +1 to post-harvest year
    for k in range(1, len(raw)):
        if raw['year'].iloc[k] == raw['year'].iloc[k-1]:
            raw['year'].iloc[k] += 1
            raw['DomAge'].iloc[k] += 1

    # save initial state of the stand
    initial_state = raw.iloc[0].copy()

    # start from 2nd row
    # raw = raw.iloc[1:]


    # combine biomasses from vallitsevat jaksot, siemenpuut etc. into one, and convert ton DM ha-1 to g C m-2
    components = ['bioTot', 'bioStemComm', 'bioStemWaste', 'bioBranches', 'bioFoliage', 'bioStumps', 'bioRootsC', 'bioRootsF']

    tmp = pd.DataFrame(data = np.zeros((len(raw), len(components))), index=raw.index, columns=components)

    for c in components:
        # find indices of columns containing substring c
        cix = [i for i, s in enumerate(cols) if c in s]
        subset = [cols[i] for i in cix]
        tmp[c] = raw[subset].sum(axis=1) * cf

    raw = pd.concat([raw, tmp], axis=1)
    
    # if simulations start from bare ground, tilavuus is zero until large trees emerge. In this case, interpolate 'vol' based on bioTot
    if CCF == False:
        vol = raw['vol'].values
        bm = raw['bioTot'].values

        ix = np.min(np.nonzero(vol[1:])) + 1
        #print(ix)
        if ix > 0:
            bb = vol[ix] / bm[ix]

            vol[1:ix] = bb * bm[1:ix]
            raw['vol'] = vol
            del ix, bb

        del vol, bm

    raw = raw[usecols]

    # interpolate linearly to annual values
    tmp = np.arange(0, max(raw['year']+1), 1)
    
    # annual motti-file
    dat = interp_to_denser_index(raw, tmp)
    dat['DomAge'] = np.floor(dat['DomAge'].values)
    
    #print(dat.head())
    #dat = dat
    #dat.iloc[0] = initial_state[usecols]
    
    # --- calculate harvest removals ---
    dat = dat.reindex(columns=dat.columns.tolist() + ['harvest_vol', 'harvest_bioFoliage', 'harvest_bioRootsF', 'harvest_bioRootsC',
                                                      'harvest_bioBranches', 'harvest_bioStumps', 'harvest_bioTot', 
                                                      'harvest_bioStemComm', 'harvest_bioStemWaste', 
                                                      'harvest_Log', 'harvest_Fibre', 
                                                      'NPP', 'FFol', 'FWD', 'CWD'])
    
    dat[['harvest_vol', 'harvest_bioFoliage', 'harvest_bioRootsF', 'harvest_bioRootsC','harvest_bioBranches', 'harvest_bioStumps', 'harvest_bioTot', 
        'harvest_bioStemComm', 'harvest_bioStemWaste', 'harvest_Log', 'harvest_Fibre', 'NPP', 'FFol', 'FWD', 'CWD']] = 0.0
    
    for k in range(1, len(dat)):
        dV = dat['vol'].iloc[k] - dat['vol'].iloc[k-1]
        if dV < 0:
            dat['harvest_vol'].iloc[k-1] = -dV
            for c in ['bioTot', 'bioStemComm', 'bioStemWaste', 'bioBranches', 'bioFoliage', 'bioStumps', 'bioRootsC', 'bioRootsF']:
                dX = dat[c].iloc[k] - dat[c].iloc[k-1]
                dat['harvest_' + c].iloc[k-1] = -dX

            # divide poistuma_runko aines between tukki and kuitu based on tukki & kuitu volume's
            sf = dat['sawVol'].iloc[k-1] / (dat['sawVol'].iloc[k-1] + dat['pulpVol'].iloc[k-1] + EPS)
            dat['harvest_Log'] = dat['harvest_bioStemComm'] * sf
            dat['harvest_Fibre'] = dat['harvest_bioStemComm'] * (1 - sf)
    
    # biomass growth (NPP, g C m-2 a-1)
    dat['NPP'] = 0.0
    for k in range(1, len(dat)):
        dX = dat['bioTot'].iloc[k] - dat['bioTot'].iloc[k-1]
        #dX = dat['hiilivarasto'].iloc[k] - dat['hiilivarasto'].iloc[k-1]
        if dX > 0:
            dat['NPP'].iloc[k] = dX
        #else:
        #    dat['NPP'].iloc[k] = np.NaN

    dat['NPP'].interpolate('linear', inplace=True)
    
    # litter inputs from harvests: foliage, FWD, CWD: this follows Ambio-paper

    dat['FFol'] = dat['harvest_bioFoliage'] + dat['harvest_bioRootsF'] 
    dat['FWD'] = dat['harvest_bioBranches'] + dat['harvest_bioRootsC']
    dat['CWD'] = dat['harvest_bioStumps'] + dat['harvest_bioStemWaste']

    #ix = np.where(np.isnan(dat['FFol']) == True)[0]
    #if len(ix) > 0:
    #    dat['FFol'].iloc[ix] = 0.0
    #    dat['FWD'].iloc[ix] = 0.0
    #    dat['CWD'].iloc[ix] = 0.0 

    data = dat.copy()
    data = data.iloc[0:-1]
    ## cycle rotations over timespan(yrs)
    
    # EAF
    if CCF == False:
        # -- rotation length--
        rl = np.max(data['DomAge'])
        
        #cycle for multiple rotations
        tmp = dat.iloc[1:-1]

        cycles = np.int(np.ceil((timespan - len(dat)) / rl))
        #print(cycles)
        for n in range(cycles):
            data = pd.concat([data, tmp])

    # CCF    
    else:
        # indices of two last harvests. We cycle data after 2nd last harvest!
        ix = np.where(dat['harvest_vol'] > 0)[0][-2:]
        tmp = dat.iloc[ix[0]+2:]
        #print(tmp)
        rl = len(tmp)
        cycles = np.int(np.ceil((timespan - len(dat)) /rl))
        
        for n in range(cycles):
            data = pd.concat([data, tmp])

    data.reset_index(inplace=True, drop=True)
    data = data.iloc[0:timespan]
    data['year'] = data.index.values

    return data
