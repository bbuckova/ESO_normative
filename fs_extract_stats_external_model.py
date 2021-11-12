import os
import glob
import pandas as pd
import numpy as np

# go through dirs in thos directory
#rootdir = '/home/barbora/Documents/Projects/Normative_Models/ESO/temp'
#save_res = '/home/barbora/Documents/Projects/Normative_Models/ESO/temp'
rootdir = '/hydradb/hydra_io/vypocty/skoch/freefs/freeSurfer/HCP_NUDZ_v7_ESO/A_FS_wo_preFS_all_links_20210907'
save_res = '/hydra-db/hydra_io/vypocty/buckova/PCN/tonda_fs_stats'
os.chdir(rootdir)

# everythiink relevant is called ESO*
# work on all visits
dirs = (glob.glob('ESO*'))

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
    wm = np.concatenate(gen_sub.iloc[:,[3]].to_numpy())# 3 means only volumes are read
    
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
# add prefixes and concatenate
wm_head = np.concatenate(wm_head.to_numpy())

head = np.append(['id'],inf_head)
head = np.concatenate([head, wm_head])
head = head[np.newaxis,:]

data_aseg = np.append(head, data, axis = 0)  
data_aseg = pd.DataFrame(data_aseg[1:], columns=data_aseg[0])
data_aseg.index=data_aseg['id']

# renaming and deleting columns to fit the datatable here:
# https://colab.research.google.com/github/saigerutherford/CPC_ML_tutorial/blob/master/tasks/1_fit_normative_models.ipynb#scrollTo=4b64f505-ad16-437a-94de-2646f35ae55f
data_aseg = data_aseg.rename(columns={"BrainSegVolNotVent":"BrainSegVolNotVentSurf", "eTIV":"EstimatedTotalIntraCranialVol"})
data_aseg = data_aseg.drop(columns=["VentricleChoroidVol", "id"])


#save_file = os.path.join(save_res,'fit_external_model_aseg_data.txt')
#data_aseg.to_csv(save_file, sep=';', index = False)


###
# Extract information from aparc.a2009s.stats
###
# pick which variable to preprocess
# 1:S. Name; 2:N. Vert.; 3:Surf Area; 4:Gray Vol.; 5:Thick. Avg.; 6:Thick. Std.; 7:Mean Curv.; 8:Gaus. Curv.; 9: Fold Ind.; 10:Curv. Ind.
suffix = ["StructName", "NumVert", "SurfArea", "GrayVol", "thickness", "ThickStd", "MeanCurv", "GausCurv", "FoldInd", "CurvInd"]

# Only extracting thicness this time
for which_var in range(5):

    for idx, idir in enumerate(dirs):
        #import pdb; pdb.set_trace()
        # extract data
        lh = pd.read_csv(os.path.join(rootdir,idir,'stats/lh.aparc.a2009s.stats'), skiprows=61, header=None, delim_whitespace=True)
        rh = pd.read_csv(os.path.join(rootdir,idir,'stats/rh.aparc.a2009s.stats'), skiprows=61, header=None, delim_whitespace=True)
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
    hdr_rh = ('rh_' + rh[0] + '_' + suffix[which_var]).to_numpy()
    hdr_lh = ('lh_' + rh[0] + '_' + suffix[which_var]).to_numpy()
    hdr = np.append(hdr_lh,hdr_rh)
    hdr = np.append(['id'], hdr)
    hdr = np.array([i.replace('_and_','&') for i in hdr])
    hdr = hdr[np.newaxis,:]

    data = np.append(hdr, data, axis = 0)
    data = pd.DataFrame(data[1:], columns=data[0])
    data_aparc = data
    data_aparc.index=data_aparc['id']
    data_aparc = data_aparc.drop(columns=['id'])


###
# Merge the two datatables and save
###
merged = data_aseg.merge(data_aparc, how="inner", on="id")
save_file = os.path.join(save_res,'fit_external_' + suffix[which_var] + '.txt')
merged.to_csv(save_file, sep=';', index = False)
    