import os
import glob
import pandas as pd
import numpy as np

# go through dirs in thos directory
#rootdir = '/home/barbora/Documents/Projects/Normative_Models/ESO/temp'
#save_res = '/home/barbora/Documents/Projects/Normative_Models/ESO/temp'
rootdir = '/freefs/freeSurfer/HCP_NUDZ_v7_ESO/A_FS_wo_preFS_all_links_20210907'
save_res = '/hydra-db/hydra_io/vypocty/buckova/PCN/tonda_fs_stats'
os.chdir(rootdir)

# everythiink relevant is called ESO*
# add _1 t filter outh other than the first visits
dirs = (glob.glob('ESO*_1'))

###
# Extract information from aseg
###
for idx, idir in enumerate(dirs):
    # load
    gen_inf = pd.read_csv(os.path.join(rootdir,idir,'stats/aseg.stats'), skiprows=14, skipfooter=90, header=None, delimiter=",",engine='python')
    # preparing general information data and header
    inf = gen_inf[3].to_numpy()

    # reading the white matter information
    gen_sub = pd.read_csv(os.path.join(rootdir,idir,'stats/aseg.stats'), skiprows=79, header=None, delim_whitespace=True)
    # preparing wm data
    wm = np.concatenate(gen_sub.iloc[:,[3,5,6]].to_numpy())
    
    # create the data_table
    aseg = np.concatenate([inf,wm])

    # extract subject id and visit
    name = idir.split('_')
    name = name[1]+'_'+name[-1]
    aseg = np.append(name,aseg)
    aseg = aseg[np.newaxis,:]
    
    if idx == 0:
        data = aseg
    else: 
        data = np.append(data, aseg, axis = 0)

# header
inf_head = gen_inf[1].str.strip().to_numpy()

# Creating wm data header
wm_head = pd.DataFrame(gen_sub.iloc[:,4])
# add copies of columns
wm_head = wm_head.assign(mean=wm_head[4])
wm_head = wm_head.assign(std=wm_head[4])
# add prefixes and concatenate
wm_head[4] = 'nvox_' + wm_head[4].astype(str)
wm_head["mean"] = 'mean_' + wm_head["mean"].astype(str)
wm_head["std"] = 'std_' + wm_head["std"].astype(str)
wm_head = np.concatenate(wm_head.to_numpy())

head = np.append(['id'],inf_head)
head = np.concatenate([head, wm_head])
head = head[np.newaxis,:]

data_aseg = np.append(head, data, axis = 0)  
data_aseg = pd.DataFrame(data_aseg[1:], columns=data_aseg[0])

save_file = os.path.join(save_res,'aseg_data.txt')
data_aseg.to_csv(save_file, sep=';', index = False)


###
# Extract information from aparc
###
# pick which variable to preprocess
# 1:S. Name; 2:N. Vert.; 3:Surf Area; 4:Gray Vol.; 5:Thick. Avg.; 6:Thick. Std.; 7:Mean Curv.; 8:Gaus. Curv.; 9: Fold Ind.; 10:Curv. Ind.
suffix = ["StructName", "NumVert", "SurfArea", "GrayVol", "ThickAvg", "ThickStd", "MeanCurv", "GausCurv", "FoldInd", "CurvInd"]

for which_var in range(2,6):

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