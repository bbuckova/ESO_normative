# Pckage that prepares and loads raw data
# load packages
import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pcntoolkit as pcn
import pickle
import scipy.stats as stats
import xarray as xr
from sklearn.model_selection import train_test_split

# importing custom functions
code_dir = ('/home/barbora/Documents/Projects/Normative_Models/ESO/code')
os.chdir(code_dir)
import clinics_desc_functions as custom
from clinics_desc_functions import prepare_data, plot_quality, trajectory_plotting, dk_roi_viz, load_clinics, en_qc, pretrained_adapt, set_seed
from temp_func import pretrained_adapt_controls

# set seed
set_seed()

# Functions
def merge_and_xarray(models_dir, file_to_merge, merged_name, indicies, idp_ids):
    """
    Merge the models into one xarray dataframe
    
    (xarray) = merge_and_xarray(models_dir, file_to_merge, merged_name, indicies, idp_ids)
    """
    
    import xarray as xr
    
    # concate and load files    
    v1 = pd.read_csv(custom.idp_concat(os.path.join(models_dir, 'V1'), file_to_merge, idp_ids, merged_name), sep=' ', index_col=0)
    v1.index = indicies

    v2 = pd.read_csv(custom.idp_concat(os.path.join(models_dir, 'V2'), file_to_merge, idp_ids, merged_name), sep=' ', index_col=0)
    v2.index = indicies

    # create xarray datasets
    xrv1 = xr.DataArray(v1[idp_ids], [('subject',list(v1.index)), ('roi', list(v1[idp_ids].columns))])
    xrv2 = xr.DataArray(v2[idp_ids], [('subject',list(v2.index)), ('roi', list(v2[idp_ids].columns))])
    xrall = xr.concat([xrv1, xrv2], pd.Index(['v1', 'v2'], name='visit'))

    return(xrall)

def only_xarray(models_dir, merged_name, indicies, idp_ids):
    xr1 = xr.DataArray(pd.read_csv(os.path.join(models_dir, 'V1', merged_name), sep=' ', index_col=0), [('subject',list(indicies)), ('roi', idp_ids)])
    xr2 = xr.DataArray(pd.read_csv(os.path.join(models_dir, 'V2', merged_name), sep=' ', index_col=0), [('subject',list(indicies)), ('roi', idp_ids)])
    xrall = xr.concat([xr1, xr2], pd.Index(['v1', 'v2'], name='visit'))
    return(xrall) 
    
def analysis_pretrained_adapt_controls(only_load=1, preproc='long', modality='thickness', test_size=0.7, pretrained='orig'):
    ###
    # where things are
    ###
    main_dir = ('/home/barbora/Documents/Projects/Normative_Models/ESO')
    data_dir = os.path.join(main_dir, 'models', 'sensitivity', 'data')

    models_dir = ('/home/barbora/Documents/Projects/Normative_Models/ESO/models/full_analysis/pretrained_zscores_'+pretrained+'_'+preproc)
    os.makedirs(models_dir, exist_ok=True)

    if pretrained=='orig':
        pretrained_dir = ('/home/barbora/Documents/Projects/Normative_Models/ESO/braincharts/models/lifespan_57K_82sites')
        model_name, site_names, site_ids_tr, idp_ids = custom.pretrained_ini(sites=82)
    elif pretrained=='nudz':
        pretrained_dir = ('/home/barbora/Documents/Projects/Normative_Models/ESO/braincharts/models/lifespan_58K_83_sites')
        model_name, site_names, site_ids_tr, idp_ids = custom.pretrained_ini(sites=83)
    
    
    images_dir = os.path.join(models_dir,'img')
    os.makedirs(images_dir, exist_ok=True)

    # Pick the preprocessing and load data
    #preproc = 'long'
    
    # Automatically loads raw QC
    v1 = pd.read_csv(os.path.join(data_dir, 'all_'+preproc+'_'+modality+'_1_qc.csv'), sep=' ', index_col=0)
    v2 = pd.read_csv(os.path.join(data_dir, 'all_'+preproc+'_'+modality+'_2_qc.csv'), sep=' ', index_col=0)

    v1_pat = v1[v1['category']=='Patient']
    v1_cont = v1[v1['category']=='Control']
    v2_pat = v2[v2['category']=='Patient']
    v2_cont = v2[v2['category']=='Control']

    site_ids_te =  sorted(set(v1_pat['site'].to_list()))


    if only_load == 0:
        ###
        # Splitting controls into adaptation and testing
        ###

        # train_test_split - split across sites, keep 30% as TESTING
        # keep the sex rate
        index_ad, index_cont_test = train_test_split(v1_cont.index, test_size = test_size, shuffle = True, random_state = 42, stratify=v1_cont['sex'])

        ###
        # Configure covariates
        ###

        # which data columns do we wish to use as covariates? 
        cols_cov = ['age','sex']

        # limits for cubic B-spline basis 
        xmin = -5 
        xmax = 110

        # Absolute Z treshold above which a sample is considered to be an outlier (without fitting any model)
        outlier_thresh = 7

        # Which visit do we wanto to analyse?
        for ivisit in range(1,3):
            print(ivisit)
            visit_dir = os.path.join(models_dir,'V'+str(ivisit))
            os.makedirs(visit_dir, exist_ok=True)

            controls_eval = 1

            if ivisit==1:
                df_ad = v1_cont.loc[index_ad] 
                df_te_cont = v1_cont.loc[index_cont_test]
                df_te_pat = v1_pat
                
                v1_cont.loc[index_ad].to_csv(os.path.join(visit_dir,'v1_cont_ad.txt'), sep=' ')
                v1_cont.loc[index_cont_test].to_csv(os.path.join(visit_dir,'v1_cont_test.txt'), sep=' ')
                v1_pat.to_csv(os.path.join(visit_dir,'v1_pat.txt'), sep=' ')

            elif ivisit == 2:
                df_ad = v1_cont.loc[index_ad] 
                df_te_cont = v2_cont.loc[index_cont_test]
                df_te_pat = v2_pat

                v2_cont.loc[index_ad].to_csv(os.path.join(visit_dir,'v2_cont_ad.txt'), sep=' ')
                v2_cont.loc[index_cont_test].to_csv(os.path.join(visit_dir,'v2_cont_test.txt'), sep=' ')
                v2_pat.to_csv(os.path.join(visit_dir,'v2_pat.txt'), sep=' ')

            # Run models
            pretrained_adapt_controls(idp_ids, site_ids_tr, site_ids_te, pretrained_dir, visit_dir, df_ad, df_te_cont, df_te_pat)
        
        ###            
        # Concatenate and save the results 
        ###
        clin_conttest = pd.read_csv(os.path.join(models_dir, 'V1', 'v1_cont_test.txt'), sep=' ', index_col=0)
        y_conttest = merge_and_xarray(models_dir, 'y_conttest.txt', 'y_conttest_merge.txt', clin_conttest.index, idp_ids)
        y_test = merge_and_xarray(models_dir, 'y_predict.txt', 'y_test_merge.txt', v1_pat.index, idp_ids)

        z_conttest = merge_and_xarray(models_dir, 'Z_conttest.txt', 'z_conttest_merge.txt', clin_conttest.index, idp_ids)
        z_test = merge_and_xarray(models_dir, 'Z_predict.txt', 'z_test_merge.txt', v1_pat.index, idp_ids)

        yhat_conttest = merge_and_xarray(models_dir, 'yhat_conttest.txt', 'yhat_conttest_merge.txt', clin_conttest.index, idp_ids)
        yhat_test = merge_and_xarray(models_dir, 'yhat_predict.txt', 'yhat_test_merge.txt', v1_pat.index, idp_ids)

        ys2_conttest = merge_and_xarray(models_dir, 'ys2_conttest.txt', 'ys2_conttest_merge.txt', clin_conttest.index, idp_ids)
        ys2_test = merge_and_xarray(models_dir, 'ys2_predict.txt', 'ys2_test_merge.txt', v1_pat.index, idp_ids)

    else:
        clin_conttest = pd.read_csv(os.path.join(models_dir, 'V1', 'v1_cont_test.txt'), sep=' ', index_col=0)
        y_conttest = only_xarray(models_dir, 'y_conttest_merge.txt', clin_conttest.index, idp_ids)
        y_test = only_xarray(models_dir, 'y_test_merge.txt', v1_pat.index, idp_ids)
        
        z_conttest = only_xarray(models_dir, 'z_conttest_merge.txt', clin_conttest.index, idp_ids)
        z_test = only_xarray(models_dir, 'z_test_merge.txt', v1_pat.index, idp_ids)

        yhat_conttest = only_xarray(models_dir, 'yhat_conttest_merge.txt', clin_conttest.index, idp_ids)
        yhat_test = only_xarray(models_dir, 'yhat_test_merge.txt', v1_pat.index, idp_ids)

        ys2_conttest = only_xarray(models_dir, 'ys2_conttest_merge.txt', clin_conttest.index, idp_ids)
        ys2_test = only_xarray(models_dir, 'ys2_test_merge.txt', v1_pat.index, idp_ids)

    # Loading and merging data
    xrv1 = xr.DataArray(v1_cont[idp_ids].loc[clin_conttest.index], [('subject',list(clin_conttest.index)), ('roi', list(v1_cont[idp_ids].columns))])
    xrv2 = xr.DataArray(v2_cont[idp_ids].loc[clin_conttest.index], [('subject',list(clin_conttest.index)), ('roi', list(v2_cont[idp_ids].columns))])
    raw_conttest = xr.concat([xrv1, xrv2], pd.Index(['v1', 'v2'], name='visit'))

    xrv1 = xr.DataArray(v1_pat[idp_ids].loc[v1_pat.index], [('subject',list(v1_pat.index)), ('roi', list(v1_pat[idp_ids].columns))])
    xrv2 = xr.DataArray(v2_pat[idp_ids].loc[v1_pat.index], [('subject',list(v1_pat.index)), ('roi', list(v2_pat[idp_ids].columns))])
    raw_test = xr.concat([xrv1, xrv2], pd.Index(['v1', 'v2'], name='visit'))


    controls = xr.concat([raw_conttest, y_conttest, yhat_conttest, ys2_conttest, z_conttest], pd.Index(['raw', 'y_orig', 'yhat_orig', 'ys2_orig', 'z_orig'], name='preproc')).to_dataset(name='features')
    patients = xr.concat([raw_test, y_test, yhat_test, ys2_test, z_test], pd.Index(['raw', 'y_orig', 'yhat_orig', 'ys2_orig', 'z_orig'], name='preproc')).to_dataset(name = 'features')

    # Adding the clinical variables
    #thick_rois = list(map(lambda x: x.replace('_thickness', ''), idp_ids))
    clin_vars = set(v1_cont.columns) - set(idp_ids)

    xrv1 = xr.DataArray(v1_pat[clin_vars], [('subject',list(v1_pat.index)), ('var', list(clin_vars))])
    xrv2 = xr.DataArray(v2_pat[clin_vars], [('subject',list(v2_pat.index)), ('var', list(clin_vars))])
    pat_clin = xr.concat([xrv1, xrv2], pd.Index(['v1', 'v2'], name='visit')).to_dataset(name = 'clinics')
    patients = xr.merge([patients, pat_clin])

    xrv1 = xr.DataArray(v1_cont[clin_vars].loc[clin_conttest.index], [('subject',list(clin_conttest.index)), ('var', list(clin_vars))])
    xrv2 = xr.DataArray(v2_cont[clin_vars].loc[clin_conttest.index], [('subject',list(clin_conttest.index)), ('var', list(clin_vars))])
    cont_clin = xr.concat([xrv1, xrv2], pd.Index(['v1', 'v2'], name='visit')).to_dataset(name = 'clinics')
    controls = xr.merge([controls, cont_clin])

    
    
    return(patients, controls)