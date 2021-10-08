import os
import glob
import pandas as pd
import numpy as np

# go through dirs in thos directory
#rootdir = '/home/barbora/Documents/Projects/Normative_Models/ESO/temp'
rootdir = '/freefs/freeSurfer/HCP_NUDZ_v7_ESO/A_FS_wo_preFS_all_links_20210907'
save_res = '/hydra-db/hydra_io/vypocty/buckova/PCN/tonda_fs_stats'
os.chdir(rootdir)

# everythiink relevant is called ESO*
# add _1 t filter outh other than the first visits
dirs = (glob.glob('ESO*_1'))

# pick which variable to preprocess
# 1:S. Name; 2:N. Vert.; 3:Surf Area; 4:Gray Vol.; 5:Thick. Avg.; 6:Thick. Std.; 7:Mean Curv.; 8:Gaus. Curv.; 9: Fold Ind.; 10:Curv. Ind.
suffix = ["StructName", "NumVert", "SurfArea", "GrayVol", "ThickAvg", "ThickStd", "MeanCurv", "GausCurv", "FoldInd", "CurvInd"]

for which_var in range(2,5):

    for idx, idir in enumerate(dirs):
        #import pdb; pdb.set_trace()
        # extract data
        rh = pd.read_csv(os.path.join(rootdir,idir,'stats/rh.aparc.stats'), skiprows=61, header=None, delim_whitespace=True)
        lh = pd.read_csv(os.path.join(rootdir,idir,'stats/lh.aparc.stats'), skiprows=61, header=None, delim_whitespace=True)
        var = np.append(rh[which_var].to_numpy(), lh[which_var].to_numpy())

        # extract subject id and visit
        name = idir.split('_')
        name = name[1]+'_'+name[-1]
        var = np.append(name,var)
        var = var[np.newaxis,:]
        
        if idx == 0:
            data = var
        else: 
            data = np.append(data, var, axis = 0)

    # create header and append to data
    hdr_rh = ('rh_' + rh[0]).to_numpy()
    hdr_lh = ('lh_' + rh[0]).to_numpy()
    hdr = np.append(hdr_rh, hdr_lh)
    hdr = np.append(['id'], hdr)
    hdr = hdr[np.newaxis,:]

    data = np.append(hdr, data, axis = 0)
    data = pd.DataFrame(data[1:], columns=data[0])
    save_file = os.path.join(save_res,'aparc_' + suffix[which_var] + '.txt')
    data.to_csv(save_file, sep=';', index = False)