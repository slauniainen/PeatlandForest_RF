Samuli Launiainen 8.10.2025

Structure of RF results (.xlsx) based on Motti-simulations and RF-calculations

sheets:

FNR - forestry on nutrient rich site
Rest_RME - restoration to open mesotrophic fen, clear-cutting at t=0 creates wood products and residues that gradually decompose
Rest_ROL - restoration to open oligotrophic fen, clear-cutting at t=0 creates wood products and residues that gradually decompose
Rest_RSM - restoration to tree-covered mire, no clear-cutting, tree C storate assumed constant in time
_Mtkg opt2% - cycled Motti-outputs used in RF calculations

year
Age	 [yr]
Vol	 [m3 ha-1]
BA	[m2 ha-1]

# carbon C stocks in the system

C_tree - tree C stock [g C m-2]
C_resid	- harvest residue (foliage, FWD, CWD) C stock, left on-site [g c m-2]
C_WP_short - short-term wood product (mean lifetime 3 yr) C stock [g C m-2]
C_WP_long - long-term wood product (mean lifetime 30 yr) C stock [g C m-2]	
C_soil - CHANGE in soil C stock SINCE year=0 [g C m-2]

# Fluxes between system components and the atmosphere. Positive sign == net emission to the atmosphere
# NOTE: CO2 fluxes in [g C m-2 a-1]
F_tree - tree NPP [g C m-2 a-1]
F_resid - emission from residues [g C m-2 a-1]
F_soil - soil emission/sink [g C m-2 a-1]
F_WP_short - emission from short-term WP [g C m-2 a-1]
F_WP_long - emission from long-term WP [g C m-2 a-1]
F_CH4 - methane emission [g CH4 m-2 a-1]
F_N2O - nitrous oxide emission[g N2O m-2 a-1]

#Radiative forcings [W m-2 (land)], represent net climate impact at time t caused by system's emissions/sinks in [0, t]	
RF_tot - total RF (=RF_totCO2 + RF_CH4 + RF_N2O)
RF_totCO2 - total RF from CO2 emissions/sinks
RF_tree	RF_resid - RF from residue CO2 emissions
RF_soil	- RF from soil net CO2 emissions/sinks
RF_WP_short	- RF from CO2 emissions from short-term WP
RF_WP_long - RF from CO2 emissions from long-term WP
RF_CH4 - RF from (soil) CH4 emissions
RF_N2O - RF from (soil) N2O emissions
