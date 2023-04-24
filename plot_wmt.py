import os
import sys
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    
    DATA_diro_FINAL= '/lcrc/group/acme/ac.cbegeman/wmt_MPASO/FINAL'
    filename = '20220126.v2.amoc.baseline.ne30pg2_EC30to60E2r2.anvil.MPASO.densityBinnedFields.0001.nc'
    with xr.open_dataset(f'{DATA_diro_FINAL}/{filename}') as ds:
       time = ds.time
       bin_centers = ds.densBinsCenters.isel(time=0)
       # The density bins are constant in time so we only need to load one
       densBinsEdges = ds.densBinsEdges.isel(time=0)
       densBinsEdge0 = densBinsEdges.isel(bnds=0)
       densBinsEdge1 = densBinsEdges.isel(bnds=1)

       # -------------------------------------------------------------------- #
       # Salinity terms
       # -------------------------------------------------------------------- #
       # timeMonthly_avg_seaIceFreshWaterFlux
       dens_IO_FWflux = ds.dens_IO_FWflux
       # Atmosphere-ocean freshwater flux at cell centers derived from
       # coupler fields. Positive into the ocean.
       dens_AO_FWflux = ds.dens_AO_FWflux
       # Sea ice-ocean and atmosphere-ocean freshwater flux at cell centers
       # derived from coupler fields. Positive into the ocean.
       dens_IOAO_FWflux = ds.dens_IOAO_FWflux
       # timeMonthly_avg_evaporationFlux
       dens_evap = ds.dens_evap
       # timeMonthly_avg_rainFlux
       dens_rain = ds.dens_rain
       # timeMonthly_avg_snowFlux
       dens_snow = ds.dens_snow
       # timeMonthly_avg_riverRunoffFlux
       dens_rivRof = ds.dens_rivRof
       # timeMonthly_avg_iceRunoffFlux
       dens_iceRof = ds.dens_iceRof
       # where(timeMonthly_avg_iceRunoffFlux < 0)
       dens_brine = ds.dens_brine
       # where(timeMonthly_avg_iceRunoffFlux > 0)
       dens_melt = ds.dens_melt

       # -------------------------------------------------------------------- #
       # Temperature terms
       # -------------------------------------------------------------------- #
       # timeMonthly_avg_shortWaveHeatFlux 
       dens_SW = ds.dens_SW
       # timeMonthly_avg_sensibleHeatFlux 
       dens_SH = ds.dens_SH
       # timeMonthly_avg_longWaveHeatFluxDown
       # + timeMonthly_avg_longWaveHeatFluxUp 
       dens_LW = ds.dens_LW
       # timeMonthly_avg_latentHeatFlux 
       dens_LH = ds.dens_LH
       # timeMonthly_avg_seaIceHeatFlux 
       dens_IH = ds.dens_IH
       # SW + SH + LWd + LWu + LH + IH
       dens_hap = ds.dens_hap

       # Annual average surface flux line plot
       fig = plt.Figure()
       # Surface freshwater flux
       dens_SFWF = dens_IOAO_FWflux.mean(dim='time')
       # Surface heat flux
       dens_SHF = dens_hap.mean(dim='time')
       dens_total = dens_SFWF + dens_SHF
       plt.plot(bin_centers, dens_SFWF, 'b', label='Surface Freshwater Flux')
       plt.plot(bin_centers, dens_SHF, 'r', label='Surface Heat Flux')
       plt.plot(bin_centers, dens_total, 'k', label='Surface Heat Flux')
       plt.savefig(f'{DATA_diro_FINAL}/ann_avg_surf_flux.png')
       plt.close(fig)
