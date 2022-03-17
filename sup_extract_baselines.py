import os
import glob
import pandas as pd
import numpy as np

# go through dirs in thos directory
#rootdir = '/home/barbora/Documents/Projects/Normative_Models/ESO/temp'
#save_res = '/home/barbora/Documents/Projects/Normative_Models/ESO/temp'
#rootdir = '/hydradb/hydra_io/vypocty/skoch/freefs/freeSurfer/HCP_NUDZ_v7_ESO/A_FS_wo_preFS_all_links_20210907'

rootdir = '/hydra/hydra_io/vypocty/skoch/HCP_NUDZ_v7_ESO/A_FS_wo_preFS_w_T2'
save_res = '/hydra-db/hydra_io/vypocty/buckova/PCN/tonda_fs_stats'

os.chdir(rootdir)

# go through correct files
import sys
visit=sys.argv[1] # prints var1
category = sys.argv[2]
preprocessing = sys.argv[3]

# everythiink relevant is called ESO*
# work on all visits
if preprocessing == 'long':
    dirs = (glob.glob('ESO_'+category+'*_'+str(visit)+'.long.*_base'))
elif preprocessing == 'cs':
    dirs = (glob.glob('ESO_'+category+'*_'+str(visit)))
else:
    print('preprocessing has to be either cs or long')

###
# Extract information from aseg
###
#for idx, idir in enumerate(dirs):
for idx, idir in enumerate(dirs):
    lh = pd.read_csv(os.path.join(rootdir,idir,'stats', 'lh.aparc.a2009s.stats'), skiprows=18, skipfooter=106, header=None, delimiter=",",engine='python') 
    rh = pd.read_csv(os.path.join(rootdir,idir,'stats', 'rh.aparc.a2009s.stats'), skiprows=18, skipfooter=106, header=None, delimiter=",",engine='python') 
    var = np.append(lh[3].to_numpy(), rh[3].to_numpy())

    name = idir.split('.')[0]

    if idx == 0:
        data = np.append(name,var[np.newaxis,:])
    else: 
        data = np.vstack([data, np.append(name,var[np.newaxis,:])])
    

hdr_rh = ('rh_' + rh[1]).to_numpy()
hdr_lh = ('lh_' + lh[1]).to_numpy()
hdr = np.append(hdr_lh,hdr_rh)
hdr = np.append(['id'], hdr)
hdr = hdr[np.newaxis,:]

data = np.append(hdr, data, axis = 0)
data = pd.DataFrame(data[1:], columns=data[0])
data_aparc = data
data_aparc.index=data_aparc['id']
data_aparc = data_aparc.drop(columns=['id'])


save_file = os.path.join(save_res,category+'_'+str(visit)+'_'+preprocessing+'rough_metrics.txt')
merged.to_csv(save_file, sep=' ', index = True)