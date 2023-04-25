import os
import glob
import sys
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    
    DATA_diro_FINAL= '/lcrc/group/acme/ac.cbegeman/wmt_MPASO/FINAL'
    regionName = "SouthernOceanBasin"
    filenames = glob.glob(f'{DATA_diro_FINAL}/*{regionName}*densityBinnedFields*.nc')
    with xr.open_dataset(filenames[0]) as dsMesh:
       # The density bins are constant in time so we only need to load one
       bin_centers = dsMesh.densBinsCenters.isel(time=0)
       densBinsEdges = dsMesh.densBinsEdges.isel(time=0)
       densBinsEdge0 = densBinsEdges.isel(bnds=0)
       densBinsEdge1 = densBinsEdges.isel(bnds=1)
       dens_bnds = np.concatenate([densBinsEdge0.values, [densBinsEdge1.values[-1]]]) 

       dens_iceRof_by_year = dsMesh.dens_iceRof.expand_dims(dim='year', axis=0)

    ds = xr.Dataset()
    data_vars = ['dens_IOAO_FWflux', 'dens_hap']
    for filename in filenames:
       with xr.open_dataset(filename) as ds_single:
           dens_iceRof_single = ds_single.dens_iceRof
           dens_iceRof_new = dens_iceRof_single.expand_dims(dim='year', axis=0)
           dens_iceRof_by_year = xr.concat([dens_iceRof_by_year, dens_iceRof_new], dim='year', join='override') 
           #ds = xr.concat([ds, ds_single], dim='time', data_vars='minimal',
           #               coords='minimal')
           for data_var in data_vars:
               if data_var in ds.keys():
                   ds = xr.concat([ds, ds_single], dim='time',
                                  data_vars={data_var},
                                  coords='minimal')
               else:
                   ds[data_var] = ds_single[data_var]

    time = ds.time

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
    # timeMonthly_avg_landIceFreshwaterFlux
    dens_ISMF = ds.dens_ISMF

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
    fig1 = plt.figure(1)
    # Surface freshwater flux
    dens_SFWF = dens_IOAO_FWflux.mean(dim='time')
    # Surface heat flux
    dens_SHF = dens_hap.mean(dim='time')
    dens_total = dens_SFWF + dens_SHF
    plt.plot(bin_centers - 1.0e3, dens_SFWF, 'b', label='Surface Freshwater Flux')
    plt.plot(bin_centers - 1.0e3, dens_SHF, 'r', label='Surface Heat Flux')
    plt.plot(bin_centers - 1.0e3, dens_total, 'k', label='Total Surface Flux')
    plt.xlabel(r'Neutral density, $\gamma_n$ ($kg\:m^{-3}$)')
    plt.ylabel('Transformation rate (Sv)')
    plt.legend()
    plt.savefig(f'{DATA_diro_FINAL}/{regionName}_ann_avg_surf_flux.png')
    plt.close(fig1)

    # Decomposition of surface FWF components
    fig2 = plt.figure(2)
    dens_EPR = dens_AO_FWflux + dens_rivRof
    plt.plot(bin_centers - 1.0e3, dens_ISMF.mean(dim='time'), 'r', label='Ice shelf melting')
    plt.plot(bin_centers - 1.0e3, dens_iceRof.mean(dim='time'), 'b', label='Sea ice formation and melting')
    plt.plot(bin_centers - 1.0e3, dens_melt.mean(dim='time'), 'orange', label='Sea ice melting')
    plt.plot(bin_centers - 1.0e3, dens_brine.mean(dim='time'), 'darkviolet', label='Sea ice formation')
    plt.plot(bin_centers - 1.0e3, dens_EPR.mean(dim='time'), 'green', label='E-P-R')
    plt.xlabel(r'Neutral density, $\gamma_n$ ($kg\:m^{-3}$)')
    plt.ylabel('Transformation rate (Sv)')
    plt.legend()
    plt.savefig(f'{DATA_diro_FINAL}/{regionName}_decomp_surf_flux.png')
    plt.close(fig2)

    fig3 = plt.figure(3)
    dens_iceRof_year_avg = dens_iceRof_by_year.mean(dim='year')
    dens_iceRof_year_avg = np.transpose(dens_iceRof_year_avg.values)
    month_bnds = np.arange(0, 13, 1)
    X, Y = np.meshgrid(month_bnds, dens_bnds)
    dens_max = np.max(np.abs(dens_iceRof_year_avg))
    plt.pcolormesh(X, Y - 1.0e3, dens_iceRof_year_avg, cmap='RdBu', vmin=-1 * dens_max, vmax=dens_max)
    plt.xlabel('Month')
    plt.ylabel(r'Neutral density, $\gamma_n$ ($kg\:m^{-3}$)')
    plt.colorbar()
    plt.savefig(f'{DATA_diro_FINAL}/{regionName}_sea_ice_monthly.png')
    plt.close(fig3)

