def prepare_models_for_analysis(patients_dir, controls_dir, analysis_dir):
    # load packages
    import os
    import glob
    import pandas as pd
    import numpy as np


    ext_scripts_dir = ('/home/barbora/Documents/Projects/Normative_Models/ESO/braincharts/scripts')
    os.chdir(ext_scripts_dir)

    from nm_utils import remove_bad_subjects, load_2d

    code_dir = ('/home/barbora/Documents/Projects/Normative_Models/ESO/code')
    os.chdir(code_dir)

    # importing custom functions
    import clinics_desc_functions as custom
    from clinics_desc_functions import prepare_data, plot_quality, trajectory_plotting, dk_roi_viz, load_clinics, en_qc, pretrained_adapt_small, set_seed

    # set seed
    set_seed()

    ######################
    # where things are
    main_dir = ('/home/barbora/Documents/Projects/Normative_Models/ESO')
    models_dir = ('/home/barbora/Documents/Projects/Normative_Models/ESO/models/zscores_comparison_long')
    os.makedirs(models_dir, exist_ok=True)
    #controls_dir = ('/home/barbora/Documents/Projects/Normative_Models/ESO/models/control_stability_long')
    #patients_dir = ('/home/barbora/Documents/Projects/Normative_Models/ESO/models/pretrained_long')
    cdata_dir = ('/home/barbora/Documents/Projects/2021_06_AZV_ESO/data')
    fsdata_dir = ('/home/barbora/Documents/Projects/Normative_Models/ESO/fs_stats')
    bdata_dir = ('/home/barbora/Documents/Projects/Normative_Models/ESO/backup/fit_external_long')
    pretrained_dir = ('/home/barbora/Documents/Projects/Normative_Models/ESO/braincharts')
    #analysis_dir = os.path.join(main_dir, 'analyses', '01_PANSS')
    images_dir = os.path.join(analysis_dir,'img')
    os.makedirs(images_dir, exist_ok=True)

    # Here we are going to load data that are already preprocessed, concatenated and so on

    # get basic parameters for pretrained models
    model_name, site_names, site_ids_tr, idp_ids = custom.pretrained_ini()

    ###
    # longitudinal paired controls
    ###
    v11_cf = custom.idp_concat(controls_dir, 'v1_Z.txt', idp_ids,  'v11_cont_z.csv', t_dir= analysis_dir)
    v12_cf = custom.idp_concat(controls_dir, 'v2_Z.txt', idp_ids,  'v12_cont_z.csv', t_dir= analysis_dir)

    # concatenate over idps and load
    v1_cont_z = pd.read_csv(v11_cf, sep = ' ', index_col=0)
    v2_cont_z = pd.read_csv(v12_cf, sep = ' ', index_col=0)

    # load original control data with clinics
    v1_cont_clin = pd.read_csv(os.path.join(controls_dir,'v1_common.csv'),index_col=0, sep = ' ', usecols=range(0,9))
    v2_cont_clin = pd.read_csv(os.path.join(controls_dir,'v2_common.csv'),index_col=0, sep = ' ', usecols=range(0,9))

    v1_cont_orig = pd.read_csv(os.path.join(controls_dir,'v1_common.csv'),index_col=0, sep = ' ')
    v2_cont_orig = pd.read_csv(os.path.join(controls_dir,'v2_common.csv'),index_col=0, sep = ' ')

    # change index before concatenation
    v1_cont_z.index = v1_cont_clin.index
    v2_cont_z.index = v2_cont_clin.index

    # concatenate
    v1_cont = pd.concat([v1_cont_clin,v1_cont_z],axis=1,join='inner')
    v2_cont = pd.concat([v2_cont_clin,v2_cont_z],axis=1,join='inner')


    ###
    # patients
    ###

    v11_pf = custom.idp_concat(os.path.join(patients_dir,'V1'), 'Z_predict.txt', idp_ids,  'v11_pat_z.csv', t_dir= models_dir)
    v12_pf = custom.idp_concat(os.path.join(patients_dir,'V2'), 'Z_predict.txt', idp_ids,  'v12_pat_z.csv', t_dir= models_dir)

    # concatenate over idps and loadscales
    v1_pat_z = pd.read_csv(v11_pf, sep = ' ', index_col=0)
    v2_pat_z = pd.read_csv(v12_pf, sep = ' ', index_col=0)

    # load clinics
    v1_pat_clin = pd.read_csv(os.path.join(patients_dir,'v1_pat.txt'), sep=' ', index_col=0, usecols=range(0,9))
    v2_pat_clin = pd.read_csv(os.path.join(patients_dir,'v2_pat.txt'), sep=' ', index_col=0, usecols=range(0,9))

    v1_pat_orig_all = pd.read_csv(os.path.join(patients_dir,'v1_pat.txt'), sep=' ', index_col=0)
    v2_pat_orig_all = pd.read_csv(os.path.join(patients_dir,'v2_pat.txt'), sep=' ', index_col=0)


    # change index before concatenation
    v1_pat_z.index = v1_pat_clin.index
    v2_pat_z.index = v2_pat_clin.index

    # concatenate
    v1_pat_all = pd.concat([v1_pat_clin,v1_pat_z],axis=1,join='inner')
    v2_pat_all = pd.concat([v2_pat_clin,v2_pat_z],axis=1,join='inner')

    # We only want to keep patients with both visits
    common = v1_pat_all.index.intersection(v2_pat_all.index)
    v1_common_id = np.where(v1_pat_all.index.isin(common))
    v2_common_id = np.where(v2_pat_all.index.isin(common))

    v1_pat = v1_pat_all.loc[common]
    v2_pat = v2_pat_all.loc[common]

    v1_pat_orig = v1_pat_orig_all.loc[common]
    v2_pat_orig = v2_pat_orig_all.loc[common]


    col_comp_id = np.where(v1_pat.columns.isin(idp_ids))[0]

    # Finally, compute the differences
    col_comp_id = np.where(v1_pat.columns.isin(idp_ids))[0]
    cont_diff = (v2_cont.iloc[:,col_comp_id]) - (v1_cont.iloc[:,col_comp_id])
    pat_diff = (v2_pat.iloc[:,col_comp_id]) - (v1_pat.iloc[:,col_comp_id])

    cont_diff.to_csv(os.path.join(analysis_dir, 'cont_diff.txt'), sep=' ', header=True, index=True)
    pat_diff.to_csv(os.path.join(analysis_dir, 'pat_diff.txt'), sep=' ', header=True, index=True)


    # Load scales - from separate excel
    scales = pd.read_excel(os.path.join(main_dir,"clinics.xlsx"))
    scales.index = scales["osobní kód (Hydra ID).1"].str.replace(r'ESO','')

    ###
    # PANSS Preparation
    ###
    # putting together PANSS score - summing across the three chapters
    P1 = [col for col in scales.columns if "PANSS 1" in col]
    P2 = [col for col in scales.columns if "PANSS 2" in col]

    v1_pat = pd.concat([v1_pat, scales[P1]], join="inner", axis=1)
    v2_pat = pd.concat([v2_pat, scales[P2]], join="inner", axis=1)

    v1_pat.rename(columns=lambda x: x.replace('PANSS/PANSS 1 / ', 'PANSS_'), inplace=True)
    v2_pat.rename(columns=lambda x: x.replace('PANSS/PANSS 2 / ', 'PANSS_'), inplace=True)

    # Rename the PANSS variables
    PANSS = [col for col in v1_pat.columns if "PANSS_" in col]

    # Create indicies of the three PANSS chapters
    PANSS_P = [col for col in v1_pat.columns if "PANSS_P" in col][0:-1]
    PANSS_N = [col for col in v1_pat.columns if "PANSS_N" in col][0:-1]
    PANSS_G = [col for col in v1_pat.columns if "PANSS_G" in col][0:-1]

    # We need to replace empy values (originally described as 'x') with NaN
    mapping = {'x': np.nan}
    v1_pat_clin = v1_pat.applymap(lambda s: mapping.get(s) if s in mapping else s)
    v2_pat_clin = v2_pat.applymap(lambda s: mapping.get(s) if s in mapping else s)

    # Create new columns that are sums of the three PANSS chapters
    v1_pat["PANSS_sumP"] = v1_pat[[col for col in v1_pat.columns if "PANSS_P" in col][0:-1]].sum(axis=1)
    v2_pat["PANSS_sumP"] = v2_pat[[col for col in v2_pat.columns if "PANSS_P" in col][0:-1]].sum(axis=1)

    v1_pat["PANSS_sumN"] = v1_pat[[col for col in v1_pat.columns if "PANSS_N" in col][0:-1]].sum(axis=1)
    v2_pat["PANSS_sumN"] = v2_pat[[col for col in v2_pat.columns if "PANSS_N" in col][0:-1]].sum(axis=1)

    v1_pat["PANSS_sumG"] = v1_pat[[col for col in v1_pat.columns if "PANSS_G" in col][0:-1]].sum(axis=1)
    v2_pat["PANSS_sumG"] = v2_pat[[col for col in v2_pat.columns if "PANSS_G" in col][0:-1]].sum(axis=1)

    ###
    # GAF
    ###
    GAF1 = [col for col in scales.columns if "Škály/GAF 1" in col]
    GAF2 = [col for col in scales.columns if "Škály/GAF 2" in col]

    v1_pat = pd.concat([v1_pat, scales[GAF1]], join="inner", axis=1)
    v1_pat.rename(columns={GAF1[0]:"GAF"}, inplace=True)

    v2_pat = pd.concat([v2_pat, scales[GAF2]], join="inner", axis=1)
    v2_pat.rename(columns={GAF2[0]:"GAF"}, inplace=True)


    # export to csv
    v1_pat.to_csv(os.path.join(analysis_dir, 'v1_pat.txt'), sep=' ', header=True, index=True)
    v2_pat.to_csv(os.path.join(analysis_dir, 'v2_pat.txt'), sep=' ', header=True, index=True)
    v1_cont.to_csv(os.path.join(analysis_dir, 'v1_cont.txt'), sep=' ', header=True, index=True)
    v2_cont.to_csv(os.path.join(analysis_dir, 'v2_cont.txt'), sep=' ', header=True, index=True)

