import sys
import os
import math
from copy import deepcopy
import numpy as np
import xarray as xr
import cftime

# FIX/NOTE/REMINDER TO MPAS-O GROUP:  NEED TO CONSIDER IMPLICATIONS OF NEUTRAL DENSITY'S ROLE
#   IN WATER MASS TRANSFORMATION CALCULATIONS 


# WATER MASS TRANSFORMATION SCRIPT DESCRIPTION:  


def xtime_to_cftime(xtime):
    year   = int(xtime[0:4])
    month  = int(xtime[5:7])
    day    = int(xtime[8:10])
    hour   = int(xtime[11:13])
    minute = int(xtime[14:16])
    second = int(xtime[17:19])
    return cftime.datetime(year, month, day, hour, minute, second, calendar='noleap')


def edit_attrs_apply_regionMask(x, x_mask, x_name=None, x_longname=None):
    """Apply regional mask and edit attributes."""
    
    x.name = x_name
    x.attrs['long_name'] = x_longname
    x = x.where(x_mask == 1, drop=True)
    return(x)



def eosstat_surface_rho_drhodt_drhods(t, s):

    """
    Compute density (rho), and density change per unit time (drhodt),
    and density change per unit salinity (drhods).
    """

    # FIX:  Use Gibbs saltwater package?  Discuss this.  Put in MPAS-tools?

    c0     =    0.0
    c1     =    1.0
    c2     =    2.0
    c3     =    3.0
    c4     =    4.0
    c5     =    5.0
    c8     =    8.0
    c10    =   10.0
    c16    =   16.0
    c1000  = 1000.0
    c10000 =10000.0
    c1p5   =    1.5
    p33    = c1/c3
    p5     = 0.500
    p25    = 0.250
    p125   = 0.125
    p001   = 0.001
    eps    = 1.0e-10
    eps2   = 1.0e-20
    bignum = 1.0e+30

    tmin = -5.0
    tmax = 50.0
    smin = 0.0
    smax = 50.0

    #-----------------------------------------------------------------------
    #  UNESCO EOS constants and JMcD bulk modulus constants
    #-----------------------------------------------------------------------

    #------ for density of fresh water (standard UNESCO)

    unt0 =   999.842594
    unt1 =  6.793952e-2
    unt2 = -9.095290e-3
    unt3 =  1.001685e-4
    unt4 = -1.120083e-6
    unt5 =  6.536332e-9

    #------ for dependence of surface density on salinity (UNESCO)

    uns1t0 =  0.824493
    uns1t1 = -4.0899e-3
    uns1t2 =  7.6438e-5
    uns1t3 = -8.2467e-7
    uns1t4 =  5.3875e-9
    unsqt0 = -5.72466e-3
    unsqt1 =  1.0227e-4
    unsqt2 = -1.6546e-6
    uns2t0 =  4.8314e-4

    #------ from Table A1 of Jackett and McDougall

    bup0s0t0 =  1.965933e+4
    bup0s0t1 =  1.444304e+2
    bup0s0t2 = -1.706103
    bup0s0t3 =  9.648704e-3
    bup0s0t4 = -4.190253e-5

    bup0s1t0 =  5.284855e+1
    bup0s1t1 = -3.101089e-1
    bup0s1t2 =  6.283263e-3
    bup0s1t3 = -5.084188e-5

    bup0sqt0 =  3.886640e-1
    bup0sqt1 =  9.085835e-3
    bup0sqt2 = -4.619924e-4

    bup1s0t0 =  3.186519
    bup1s0t1 =  2.212276e-2
    bup1s0t2 = -2.984642e-4
    bup1s0t3 =  1.956415e-6

    bup1s1t0 =  6.704388e-3
    bup1s1t1 = -1.847318e-4
    bup1s1t2 =  2.059331e-7
    bup1sqt0 =  1.480266e-4
    bup2s1t1 =  6.128773e-8
    bup2s1t2 =  6.207323e-10

    #=================================================
    # To prevent problems with garbage on land points or ghost cells
    #=================================================
    TQ = t.where(t < tmax, other=tmax)      # this will preserves metadata
    TQ = TQ.where(TQ > tmin, other=tmin)
    
    SQ = s.where(s < smax, other=smax)
    SQ = SQ.where(SQ > smin, other=smin)

    p   = c0
    p2  = p*p
    SQR = np.sqrt(SQ)
    T2  = TQ*TQ


    #=================================================
    # Calculate surface (p=0) values from UNESCO eqns.
    #=================================================

    WORK1 = uns1t0 + uns1t1*TQ + \
           (uns1t2 + uns1t3*TQ + uns1t4*T2)*T2
    WORK2 = SQR*(unsqt0 + unsqt1*TQ + unsqt2*T2)

    RHO_S = unt1*TQ + (unt2 + unt3*TQ + (unt4 + unt5*TQ)*T2)*T2 \
                    + (uns2t0*SQ + WORK1 + WORK2)*SQ

    rho   = unt0 + RHO_S

    drhodt = unt1 + c2*unt2*TQ +                      \
             (c3*unt3 + c4*unt4*TQ + c5*unt5*T2)*T2 + \
             (uns1t1 + c2*uns1t2*TQ +                 \
             (c3*uns1t3 + c4*uns1t4*TQ)*T2 +          \
             (unsqt1 + c2*unsqt2*TQ)*SQR )*SQ

    drhods  = c2*uns2t0*SQ + WORK1 + c1p5*WORK2

    return rho, drhodt, drhods    





if __name__ == "__main__":
    
    # FIX:  CHECK DOCSTRING
#	"""
#    Process MPASO output fields and compute derived fields, masked versions of both
#    native and derived fields, and write the fields to intermediate files for later use
#    as part of surface water mass transformation calculations.
#    
#
#    Keywords:  water mass transformation, mask
#	
#	"""

    
    # ----------------------------------------------------------------------
    # USER INPUTS -- FIX: DO THIS THROUGH NAMELIST?
    yrBeg = 1
    yrEnd = 2
    caseName = "20220126.v2.amoc.baseline.ne30pg2_EC30to60E2r2.anvil"
    
    # Location where MPAS-O "timeSeriesStatsMonthly" exist
    DATA_dir = f'/lcrc/group/e3sm/ac.abarthel/E3SMv2/{caseName}/archive/ocn/hist'
    
    # Define any MPAS-O restart file that uses the same grid as the data being processed (ideally, a restart file from the run being analyzed)
    #   This file is ONLY used to retrieve latCell, lonCell, and areaCell associated with caseName
    DATA_grid_file = f'/lcrc/group/e3sm/ac.abarthel/E3SMv2/{caseName}/archive/rest/0003-01-01-00000/{caseName}.mpaso.rst.0003-01-01_00000.nc'
    
    # Directory where intermediate files will be written
    DATA_diro_step1= '/lcrc/group/acme/ac.cbegeman/wmt_MPASO/step1_data'
    
    # Directory where final output files will be written
    DATA_diro_FINAL= '/lcrc/group/acme/ac.cbegeman/wmt_MPASO/FINAL'
    
    # File containing density bin edges
    DATA_densBins = '/lcrc/group/acme/ac.benedict/forCarolyn/Rho.400.nc'
    
    # File containing region masks
    #   It is expected that the mask file uses the same horizontal grid as the run being
    #     analyzed, and that mask values are 1 within the region of interest and 0 otherwise.
    MASK_file = "/lcrc/group/acme/ac.benedict/forCarolyn/EC30to60E2r2_North_Atlantic_Subpolar_v2.nc"

    # Define region to analyze
    regionNameUse = "Greenland Sea"  # if running via driver bash script:  example:  "Greenland Sea"   FIX:  This should be user input
    regionNameTag = "Greenland"  # if running via driver bash script:  example:  "Greenland"
#    regionNameUse = getenv("REGIONNAMEUSE")  # if running via driver bash script:  example:  "Greenland Sea"
#    regionNameTag = getenv("REGIONNAMETAG")  # if running via driver bash script:  example:  "Greenland"
    # ----------------------------------------------------------------------
    
    print("\n\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f'regionNameUse: {regionNameUse}')
    print(f'regionNameTag: {regionNameTag}')
    
    
    # If output directories do not exist, create them
    try:
      os.makedirs( DATA_diro_step1, exist_ok=True )
    except FileExistsError:
      pass
    try:
      os.makedirs( DATA_diro_FINAL, exist_ok=True )
    except FileExistsError:
      pass
    
    
    # Open mask file, read mask, identify user-selected region of interest
    with xr.open_dataset(MASK_file) as ds_Mask:
      regionNamesALL = ds_Mask['regionNames'].values     # returns (nRegions)
      regionNamesALL = [str(region, 'utf-8') for region in regionNamesALL]   # Clobbers original regionNamesALL
      regionIndex = regionNamesALL.index(regionNameUse)
      print("\nWMT: Available mask region names:")
      for region in regionNamesALL:
        print(region)
      print(f'WMT: Selected region is: {regionNamesALL[regionIndex]} (index={regionIndex})')
      regCellMask = ds_Mask['regionCellMasks'].isel(nRegions=regionIndex)   # returns (nCells)... Expecting 1 within specified region, 0 otherwise
      
      # (The following was originally done in "4.DensBin.LatLonArea.NATL.ncl" but can be done now instead.)
      regionVertexMask = ds_Mask['regionVertexMasks'].isel(nRegions=regionIndex)
    
    
    # Open any MPAS-O restart file that uses the same grid as the data being processed (ideally, a restart file from the run being analyzed)
    with xr.open_dataset(DATA_grid_file) as DATA_rr:
      print(f'Retrieving MPAS-O grid information:  DATA_grid_file: {DATA_grid_file}')
      lonCell        = np.rad2deg(DATA_rr['lonCell'])
      latCell        = np.rad2deg(DATA_rr['latCell'])
      areaCell       = DATA_rr['areaCell']
#      lonVertex      = np.rad2deg(DATA_rr['lonVertex'])     # FIX:  remove these, not really needed
#      latVertex      = np.rad2deg(DATA_rr['latVertex'])
#      verticesOnCell = DATA_rr['verticesOnCell']
#      nEdgesOnCell   = DATA_rr['nEdgesOnCell']
#      bottomDepth    = DATA_rr['bottomDepth']


    # Mask grid variables to specified region
    #   This was originally done in NCL script 4.DensBin.LatLonArea.NATL.ncl but can be
    #   done here instead, outside of the YR (year) and MO (month) loops below.  The
    #   following are versions of lonCell, latCell, and areaCell in which the regional
    #   mask has been applied and cells external to the region of interest have been
    #   dropped/removed.
#    lonCelln  = lonCell.where(regCellMask,drop=True)
#    latCelln  = latCell.where(regCellMask,drop=True)
    areaCelln = areaCell.where(regCellMask,drop=True)
        # areaCelln is needed for original "step 5".  It represents the region-masked
        #   cell area values and is used as a multiplier in step 5 (originally, script
        #   5.sumWMT.NATL.ncl)
    
    
    # -------    Defining density bin centers and edges    --------
    # Read file containing density bin edges
    with xr.open_dataset(DATA_densBins) as ds_densBins:
      print(f'Retrieving density bin information:  DATA_densBins: {DATA_densBins}')
      densBinsEdges = ds_densBins['rho']
      drho          = densBinsEdges[1:] - densBinsEdges[0:len(densBinsEdges)-1]
      
    # Create an array that contains densBinsCenters (bin centers)
    densBinsCenters = 0.5 * (densBinsEdges[0:-1] + densBinsEdges[1:])
    densBinsCenters = densBinsCenters.rename({'lev': 'bin'})
    
    # Create CF-standard 2D bin edge array -- this essential remaps densBinsEdges
    #   from 1D to 2D
    densBinsEdgesOut = np.zeros( (len(densBinsCenters),2), dtype=type(densBinsCenters) )
    densBinsEdgesOut[:,0] = densBinsEdges[0:-1]
    densBinsEdgesOut[:,1] = densBinsEdges[1:]
    
    print('\nDensity bin structure TEST PRINT:')
    print("%6s    %14s    | %14s %14s %14s    | %14s" % ("", "Bin low edge","Bin low edge","Bin center","Bin hi edge","Bin hi edge"))
    for i in range(len(densBinsCenters)):
      print(f'{i:6d}    {densBinsEdgesOut[i,0]:14.5f}    | {densBinsEdges[i]:14.5f} {densBinsCenters[i]:14.5f} {densBinsEdges[i+1]:14.5f}    | {densBinsEdgesOut[i,1]:14.5f}')
    print('\n\n')
    #sys.exit()
    
    
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Read/compute variables and apply mask, looping over months     
    for YR in range(yrBeg,yrEnd+1):
        YYYY = f'{YR:04d}'
        
        dsOuter   = []
        timeStart = []
        timeEnd   = []
        
        for MO in range(1,12+1):
            MM = f'{MO:02d}'
            
            # Open MPAS-O timeSeriesStatsMonthly file
            f_in0  = f'{DATA_dir}/{caseName}.mpaso.hist.am.timeSeriesStatsMonthly.{YYYY}-{MM}-01.nc'
            print(f'Reading: {f_in0}')
            ds_in0 = xr.open_dataset(f_in0)
            
            # ---  Time  ---------------------------------
            xtime_startMonthly = str(ds_in0["xtime_startMonthly"].values, 'utf-8')
            xtime_endMonthly   = str(ds_in0["xtime_endMonthly"].values, 'utf-8')
            timeCurrentMonthStart = xtime_to_cftime(xtime_startMonthly)   # !!! NOTE !!!:  This assumes calendar='noleap'
            timeCurrentMonthEnd   = xtime_to_cftime(xtime_endMonthly)     # !!! NOTE !!!:  This assumes calendar='noleap'
            #print(timeCurrentMonthStart)
            #print(timeCurrentMonthEnd)
            timeCurrentMonthStart_xr = xr.DataArray(data=timeCurrentMonthStart, name="timeCurrentMonthStart_xr")
            timeCurrentMonthEnd_xr   = xr.DataArray(data=timeCurrentMonthEnd, name="timeCurrentMonthEnd_xr")

            
            # ---  Surface temperature and salinity  ---------------------------------
            temp   = ds_in0["timeMonthly_avg_activeTracers_temperature"].isel(Time=0, nVertLevels=0)   # Expected input is (Time, nCells, nLevels)
            salt   = ds_in0["timeMonthly_avg_activeTracers_salinity"].isel(Time=0, nVertLevels=0)
            
            # Mask input variables to user-selected region
            #   Use 'drop' option to remove cells outside of interest region to reduce memory/storage (also consistent with original WMT scripts)
            temp = temp.where(regCellMask == 1, drop=True)
            salt = salt.where(regCellMask == 1, drop=True)

            

            # ---  Surface heat flux  ---------------------------------
            SW   = ds_in0["timeMonthly_avg_shortWaveHeatFlux"].isel(Time=0)    # Expected input is (Time, nCells)
            SH   = ds_in0["timeMonthly_avg_sensibleHeatFlux"].isel(Time=0)     # Expected input is (Time, nCells)
            LWd  = ds_in0["timeMonthly_avg_longWaveHeatFluxDown"].isel(Time=0) # Expected input is (Time, nCells)
            LWu  = ds_in0["timeMonthly_avg_longWaveHeatFluxUp"].isel(Time=0)   # Expected input is (Time, nCells)
            LH   = ds_in0["timeMonthly_avg_latentHeatFlux"].isel(Time=0)       # Expected input is (Time, nCells)
            IH   = ds_in0["timeMonthly_avg_seaIceHeatFlux"].isel(Time=0)       # Expected input is (Time, nCells)
            
            # Derived fields
            LW = LWd + LWu
            LW.attrs = LWd.attrs
            hap = SW + SH + LWd + LWu + LH + IH
            swsub = SW * 0.0558      # FIX:  What is this?  Check this.
            hap = hap + swsub
            hap.attrs = SW.attrs
            
            # Mask input variables to user-selected region
            #   Use 'drop' option to remove cells outside of interest region to reduce memory/storage (also consistent with original WMT scripts)
            SW  = SW.where(regCellMask == 1, drop=True)
            SH  = SH.where(regCellMask == 1, drop=True)
            LW  = edit_attrs_apply_regionMask(LW, regCellMask, \
                                                  'timeMonthly_avg_longWaveHeatFluxNet', \
                                                  'long wave heat flux at cell centers from coupler. Positive into the ocean.')
            LH  = LH.where(regCellMask == 1, drop=True)
            IH  = IH.where(regCellMask == 1, drop=True)
            hap  = edit_attrs_apply_regionMask(hap, regCellMask, \
                                                  'timeMonthly_avg_HeatFluxNet', \
                                                  'Net surface heat flux. Positive into the ocean')     # Fix:  have oceanographer confirm that 'hap' is accurately characterized as net sfc heat flx

            
            # ---  Surface freshwater flux  ---------------------------------
            evap    = ds_in0["timeMonthly_avg_evaporationFlux"].isel(Time=0)      # Expected input is (Time, nCells)
            rain    = ds_in0["timeMonthly_avg_rainFlux"].isel(Time=0)             # Expected input is (Time, nCells)
            snow    = ds_in0["timeMonthly_avg_snowFlux"].isel(Time=0)             # Expected input is (Time, nCells)
            rivRof  = ds_in0["timeMonthly_avg_riverRunoffFlux"].isel(Time=0)      # Expected input is (Time, nCells)
            iceRof  = ds_in0["timeMonthly_avg_iceRunoffFlux"].isel(Time=0)        # Expected input is (Time, nCells)
            IO      = ds_in0["timeMonthly_avg_seaIceFreshWaterFlux"].isel(Time=0) # Expected input is (Time, nCells)
            
            # Derived fields
            brine = iceRof.where(iceRof < 0., other=0.)  # confirmed that this preserves metadata
            melt = iceRof.where(iceRof > 0., other=0.)
            AO = evap + rain + snow + rivRof + iceRof
            IOAO = AO + IO
            
            # Mask input variables to user-selected region
            #   Use 'drop' option to remove cells outside of interest region to reduce memory/storage (also consistent with original WMT scripts)
            brine  = edit_attrs_apply_regionMask(brine, regCellMask, 'brine', 'brine')
            melt   = edit_attrs_apply_regionMask(melt, regCellMask, 'melt', 'melt')
            AO     = edit_attrs_apply_regionMask(AO, regCellMask, 'AO_FreshWaterFlux', \
                                                'Atmosphere-ocean freshwater flux at cell centers derived from coupler fields. Positive into the ocean.')
            IOAO  = edit_attrs_apply_regionMask(IOAO, regCellMask, 'IO_AO_FreshWaterFlux', \
                                                'Sea ice-ocean and atmosphere-ocean freshwater flux at cell centers derived from coupler fields. Positive into the ocean.')
            IO     = IO.where(regCellMask == 1, drop=True)
            evap   = evap.where(regCellMask == 1, drop=True)
            rain   = rain.where(regCellMask == 1, drop=True)
            snow   = snow.where(regCellMask == 1, drop=True)
            rivRof = rivRof.where(regCellMask == 1, drop=True)
            iceRof = iceRof.where(regCellMask == 1, drop=True)
            
            
            # ---  Write masked data to intermediate file  ---------------------------------
            
            # Convert dataArray to xArray *dataset*
            # Note:  dsMSE = mse.to_dataset does NOT work, see:
            #  https://stackoverflow.com/questions/72046736/xarrays-to-dataset-and-global-attributes
            dsOut = xr.Dataset()
            
            # State variables
            dsOut['temp'] = temp     # This should also write metadata to dsOut
            dsOut['salt'] = salt
            
            # Heat fluxes
            dsOut['SW'] = SW
            dsOut['SH'] = SH
            dsOut['LW'] = LW
            dsOut['LH'] = LH
            dsOut['IH'] = IH
            dsOut['hap'] = hap
            
            #Freshwater fluxes
            dsOut['brine'] = brine
            dsOut['melt'] = melt
            dsOut['IO_FWflux'] = IO
            dsOut['AO_FWflux'] = AO
            dsOut['IOAO_FWflux'] = IOAO
            dsOut['evap'] = evap
            dsOut['rain'] = rain
            dsOut['snow'] = snow
            dsOut['rivRof'] = rivRof
            dsOut['iceRof'] = iceRof
            
            dsOut.to_netcdf(path=f'{DATA_diro_step1}/{caseName}.MPASO.masked.T_S_HF_FW.{YYYY}{MM}.nc')
            print(f'Saving intermediate file: {DATA_diro_step1}/{caseName}.MPASO.masked.T_S_HF_FW.{YYYY}{MM}.nc')
            
            dsOut.close()
            ds_in0.close()   # Also close original input data file, no longer needed 
            
            
            # ---  Delete variables for next loop iteration  ------------------
            # Do not need to delete each variable within loop, in-loop variables will
            #   be clobbered when redefined.


            
            # =====================================================================================
            # * * * * *   THE FOLLOWING CODE WAS ORIGINALLY IN 3.Calc.densf.NATL.py   * * * * * * *
            
            # Load intermediate dataset
            fIn  = f'{DATA_diro_step1}/{caseName}.MPASO.masked.T_S_HF_FW.{YYYY}{MM}.nc'
            dsIn = xr.open_dataset(fIn)
            
            rUnit2mass      = 1.035e+03
            HeatCapacity_cp = 3.994e+03
            
            fluxfac_T  = 1.0/rUnit2mass/HeatCapacity_cp
            fluxfac_S  = 1.0/rUnit2mass
            
            # Compute density and rates of change of density
            dens, drhodt, drhods = eosstat_surface_rho_drhodt_drhods(temp, salt)
            
            # Read temperature into memory, then drop temperature from dsIn and apply
            #   density masking in bulk to remaining variables
            temp = dsIn['temp'].compute     # Read into memory
            dsIn = dsIn.drop_vars('temp')   # FIX:  Check if dsIn is not None
            dens = dens.where(dens > 1000.5, other=np.nan)    # dens must be masked separately, it is not in dsIn
            dsIn = dsIn.where(dens > 1000.5, other=np.nan)    # Apply density masking to all variables in dsIn (excluding temp)
            
            # Define salinity and temperature factors
            salt_factor = -1. * drhods * fluxfac_S * dsIn.salt    # The -1 changes the surface flux sign convention in equations below
            temp_factor = drhodt * fluxfac_T
            
            # Create empty xArray dataSet to store density-related variables
            dsOutDens = xr.Dataset()
            
            # Compute salinity terms   FIX: have oceanographer define long_names for these terms?
            for varName in ['IO_FWflux','brine','melt','AO_FWflux','IOAO_FWflux','evap','rain','snow','rivRof','iceRof' ]:
                dsOutDens[f'dens_{varName}'] = salt_factor * dsIn[varName]
            
            # Compute temperature terms   FIX: have oceanographer define long_names for these terms?
            for varName in ['SW','SH','LW','LH','IH','hap']:
                dsOutDens[f'dens_{varName}'] = temp_factor * dsIn[varName]
            
            # FIX:  The units and long_names of the terms in dsOutDens are missing at this
            #       point -- they should be populated.  The dimension name is correct ('nCells')
            #       but there is no nCells coordinate variable (this may be okay)
            
            
            
            
            
            
            # =====================================================================================
            # NOTE:  At this point we have available to us the following:
            #
            # 1. dsIn:  xArray dataSet containing regionally and density masked versions of
            #           state variables (temp, salt), heat fluxes, and freshwater fluxes.
            #           These fields have full metadata.  Note that 'temp' is no longer
            #           included in dsIn.
            # 2. temp:  xArray dataArray containing regionally masked ocean temperature,
            #           with full metadata.
            # 3. dsOutDens:  (Identical to contents of "densf" file from original NCL scripts)
            #                xArray dataSet containing regionally masked salinity and
            #                temperature terms (FIX: An oceanographer needs to better
            #                define this).  These fields have no attributes.
            
            
            
            
            
            
            # =====================================================================================
            # * * * * *   THE FOLLOWING CODE WAS ORIGINALLY IN 4.DensBin.LatLonArea.NATL.ncl   * * * * * * *
            
            # Script 4.DensBin.LatLonArea.NATL.ncl was designed to mask grid variables
            #   (latCell, lonCell, areaCell, plus others) using the mask file.  I have
            #   done this masking above, outside of the YR (year) and MO (month) loops.
            #   Xylar and I agreed that the bottomDepth, latVertex, lonVertex, and
            #   nEdgesOnCell need not be masked or written to file.
            
            
            
            
            
            
            
            # =====================================================================================
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
            #             THE FOLLOWING CODE WAS ORIGINALLY IMPLEMENTED IN
            #             4.DensBin.MPASO.new.timeWind1.ncl
            #             AND
            #             5.sumWMT.NATL.ncl
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
            
            # Script 4.DensBin.MPASO.new.timeWind1.ncl bins all flux fields (not state
            #   fields) by density thresholds read from a file.  The density bin edges
            #   are read above, outside of the YR (year) and MO (month) loops.  All of the
            #   fields (except for temp and salt) are retained within the region of interest
            #   and set to zero outside of this region.
            #
            # Script 5.sumWMT.NATL.ncl takes the density-binned, regionally-masked fields
            #   and sums them and and divides by two factors:  1.e+6 and drho 
            
            # Multiply each variable within dsOutDens dataset by the areas of each cell.
            #   Each variable is one-dimensional (and the dimension is a subset of nCells
            #   that correspond to the selected spatial region) and areaCelln is also 1D
            #   and should represent the cell area values of the corresponding selected
            #   region. 
            dsOutDens = dsOutDens * areaCelln
            
            dsInner = []     # List to hold processed dataSets spanning density bins only
            for ib in range(len(densBinsEdges)-1):
                maskDens = np.logical_and(dens >= densBinsEdges[ib], dens < densBinsEdges[ib+1])     # Use 'densBinsEdges' here to follow what Hyein did
                dsOutDensMaskSum = dsOutDens.where(maskDens).sum(dim='nCells') / (1.e+6 * drho[ib])  # dsOutDensMaskSum is a scalar (sum is over nCells, for current density bin)
                
                dsInner.append(dsOutDensMaskSum)   # At the end of ib loop, this will be of size len(densBinsCenters)
              
            
            # --------------------------
            # Outside of ib (density bin) loop
            dsInner = xr.concat(dsInner, dim='bin')     # FIX:  Not sure how dsInner (or xr.concat) knows what dim='bin' is...?  Or are we -imposing- the dimension name?
            dsInner['densBinsEdges']   = (('bin', 'bnds'), densBinsEdgesOut)
            dsInner['densBinsCenters'] = densBinsCenters
#            print('\n')
#            print(dsInner)
            
            dsOuter.append(dsInner)    # This is clobbered whenever there is a new year (above, just before we enter MO/month loop)... 
                                       # ...so at the end of each year this will be size (12,len(densBinsCenters))
            timeStart.append(timeCurrentMonthStart_xr)
            timeEnd.append(timeCurrentMonthEnd_xr)
            del timeCurrentMonthStart_xr,timeCurrentMonthEnd_xr
        
        # --------------------------
        # Outside of MO (month) loop
#        print('\n')
#        print(timeStart)
#        print('\n')
#        print(timeEnd)
        
        timeStart = xr.concat(timeStart, dim='time')
        timeEnd   = xr.concat(timeEnd, dim='time')
        
#        print('\n')
#        print(timeStart)
#        print('\n')
#        print(timeEnd)
#        #sys.exit()
        
        time_bnds = np.zeros( (12,2), dtype=type(timeCurrentMonthStart) )
        time_bnds[:,0] = timeStart[:]
        time_bnds[:,1] = timeEnd[:]
        
        dsOuter = xr.concat(dsOuter, dim='time')      # FIX:  Not sure how dsInner (or xr.concat) knows what dim='bin' is...?  Or are we -imposing- the dimension name?
        dsOuter['time_bnds'] = (('time', 'bnds'), time_bnds)
        dsOuter['time'] = timeStart
        
          
        # Write (12,399,2) data to file
        dsOuter.to_netcdf(path=f'{DATA_diro_FINAL}/{caseName}.MPASO.densityBinnedFields.{YYYY}.nc')
        print(f'\nSaving FINAL file: {DATA_diro_FINAL}/{caseName}.MPASO.densityBinnedFields.{YYYY}.nc')
        dsOut.close()    
    
    # --------------------------      
    # end of YR (year) loop



        
# Issues to consider:
# (0) Test adding drop to dsOutDens.where(maskDens) (L515)
# (1) Can we apply climatology first?  Likely not.
# (2) Plots -- line colors

# To do:
# Finish the script
# Test vs. NCL
# Plot
              

