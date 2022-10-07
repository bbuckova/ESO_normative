import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import random

from pcntoolkit.util.utils import compute_MSLL, create_design_matrix

code_dir = ('/home/barbora/Documents/Projects/Normative_Models/ESO/code')
os.chdir(code_dir)
from temp_func_pcn import estimate, predict, evaluate

def pretrained_adapt_controls(idp_ids, site_ids_tr, site_ids_te, pretrained_dir, visit_dir, df_ad, df_tec, df_te):
    """
    pretrained_adapt(idp_ids, site_ids_tr, site_ids_te, pretrained_dir, visit_dir, df_ad, df_te)
    """
    # which data columns do we wish to use as covariates? 
    cols_cov = ['age','sex']

    # limits for cubic B-spline basis 
    xmin = -5 
    xmax = 110

    # Absolute Z treshold above which a sample is considered to be an outlier (without fitting any model)
    outlier_thresh = 7

    for idp_num, idp in enumerate(idp_ids): 
        print('Running IDP', idp_num, idp, ':')
        idp_dir = os.path.join(pretrained_dir,'models','lifespan_57K_82sites', idp)
        idp_visit_dir = os.path.join(visit_dir,idp)
        os.makedirs(idp_visit_dir, exist_ok=True)
        os.chdir(idp_visit_dir)
        
        # extract and save the response variables for the test set
        y_tec = df_tec[idp].to_numpy()
        y_te = df_te[idp].to_numpy()

        # save the variables
        resp_file_tec = os.path.join(idp_visit_dir, 'resp_tec.txt') 
        resp_file_te = os.path.join(idp_visit_dir, 'resp_te.txt') 
        
        np.savetxt(resp_file_tec, y_tec)
        np.savetxt(resp_file_te, y_te)
            
        # configure and save the design matrix
        cov_file_tec = os.path.join(idp_visit_dir, 'cov_bspline_tec.txt')
        cov_file_te = os.path.join(idp_visit_dir, 'cov_bspline_te.txt')
        
        X_te = create_design_matrix(df_te[cols_cov], 
                                    site_ids = df_te['site'],
                                    all_sites = site_ids_tr,
                                    basis = 'bspline', 
                                    xmin = xmin, 
                                    xmax = xmax)
        np.savetxt(cov_file_te, X_te)
        
        X_tec = create_design_matrix(df_tec[cols_cov], 
                                    site_ids = df_tec['site'],
                                    all_sites = site_ids_tr,
                                    basis = 'bspline', 
                                    xmin = xmin, 
                                    xmax = xmax)
        np.savetxt(cov_file_tec, X_tec)
        
        # check whether all sites in the test set are represented in the training set
        if all(elem in site_ids_tr for elem in site_ids_te):
            print('All sites are present in the training data')
            
            # just make predictions
            y, yhat_te, s2_te, Z = predict(cov_file_te, 
                                        alg='blr', 
                                        respfile=resp_file_te, 
                                        model_path=os.path.join(idp_dir,'Models'))
        else:
            print('Some sites missing from the training data. Adapting model')
            
            # save the covariates for the adaptation data
            X_ad = create_design_matrix(df_ad[cols_cov], 
                                        site_ids = df_ad['site'],
                                        all_sites = site_ids_tr,
                                        basis = 'bspline', 
                                        xmin = xmin, 
                                        xmax = xmax)
            cov_file_ad = os.path.join(idp_visit_dir, 'cov_bspline_ad.txt')          
            np.savetxt(cov_file_ad, X_ad)
            
            # save the responses for the adaptation data
            resp_file_ad = os.path.join(idp_visit_dir, 'resp_ad.txt') 
            y_ad = df_ad[idp].to_numpy()
            np.savetxt(resp_file_ad, y_ad)
        
            # save the site ids for the adaptation data
            sitenum_file_ad = os.path.join(idp_visit_dir, 'sitenum_ad.txt') 
            site_num_ad = df_ad['sitenum'].to_numpy(dtype=int)
            np.savetxt(sitenum_file_ad, site_num_ad)
            
            # save the site ids for the test data 
            sitenum_file_tec = os.path.join(idp_visit_dir, 'sitenum_tec.txt')
            site_num_tec = df_tec['sitenum'].to_numpy(dtype=int)
            np.savetxt(sitenum_file_tec, site_num_tec)
            
            sitenum_file_te = os.path.join(idp_visit_dir, 'sitenum_te.txt')
            site_num_te = df_te['sitenum'].to_numpy(dtype=int)
            np.savetxt(sitenum_file_te, site_num_te)

            # adaptation files are among inputs to adjust the offset 
            y, yhat_tec, s2_tec, Z = predict(cov_file_tec, 
                                        alg = 'blr', 
                                        respfile = resp_file_tec, 
                                        model_path = os.path.join(idp_dir,'Models'),
                                        adaptrespfile = resp_file_ad,
                                        adaptcovfile = cov_file_ad,
                                        adaptvargroupfile = sitenum_file_ad,
                                        testvargroupfile = sitenum_file_tec,
                                        outputsuffix = 'cont_test')
            
            y, yhat_te, s2_te, Z = predict(cov_file_te, 
                                        alg = 'blr', 
                                        respfile = resp_file_te, 
                                        model_path = os.path.join(idp_dir,'Models'),
                                        adaptrespfile = resp_file_ad,
                                        adaptcovfile = cov_file_ad,
                                        adaptvargroupfile = sitenum_file_ad,
                                        testvargroupfile = sitenum_file_te)      

            # computation of model-specific and data-specific noise
            #with open(os.path.join(idp_dir,'Models', 'NM_0_0_estimate.pkl'), 'rb') as handle:
            #    nm = pickle.load(handle)
            # extract the different variance components to visualise
            #beta, junk1, junk2 = nm.blr._parse_hyps(nm.blr.hyp, X_dummy)
            #s2n = 1/beta # variation (aleatoric uncertainty)
            #s2s = s2-s2n # modelling uncertainty (epistemic uncertainty)


def pretrained_adapt(idp_ids, site_ids_tr, site_ids_te, pretrained_dir, visit_dir, df_ad, df_te):
    """
    pretrained_adapt(idp_ids, site_ids_tr, site_ids_te, pretrained_dir, visit_dir, df_ad, df_te)
    """
    # which data columns do we wish to use as covariates? 
    cols_cov = ['age','sex']

    # limits for cubic B-spline basis 
    xmin = -5 
    xmax = 110

    # Absolute Z treshold above which a sample is considered to be an outlier (without fitting any model)
    outlier_thresh = 7

    for idp_num, idp in enumerate(idp_ids): 
        print('Running IDP', idp_num, idp, ':')
        idp_dir = os.path.join(pretrained_dir,'models','lifespan_57K_82sites', idp)
        idp_visit_dir = os.path.join(visit_dir,idp)
        os.makedirs(idp_visit_dir, exist_ok=True)
        os.chdir(idp_visit_dir)
        
        # extract and save the response variables for the test set
        y_te = df_te[idp].to_numpy()
        
        # save the variables
        resp_file_te = os.path.join(idp_visit_dir, 'resp_te.txt') 
        np.savetxt(resp_file_te, y_te)
            
        # configure and save the design matrix
        cov_file_te = os.path.join(idp_visit_dir, 'cov_bspline_te.txt')
        X_te = create_design_matrix(df_te[cols_cov], 
                                    site_ids = df_te['site'],
                                    all_sites = site_ids_tr,
                                    basis = 'bspline', 
                                    xmin = xmin, 
                                    xmax = xmax)
        np.savetxt(cov_file_te, X_te)
        
        # check whether all sites in the test set are represented in the training set
        if all(elem in site_ids_tr for elem in site_ids_te):
            print('All sites are present in the training data')
            
            # just make predictions
            y, yhat_te, s2_te, Z = predict(cov_file_te, 
                                        alg='blr', 
                                        respfile=resp_file_te, 
                                        model_path=os.path.join(idp_dir,'Models'))
        else:
            print('Some sites missing from the training data. Adapting model')
            
            # save the covariates for the adaptation data
            X_ad = create_design_matrix(df_ad[cols_cov], 
                                        site_ids = df_ad['site'],
                                        all_sites = site_ids_tr,
                                        basis = 'bspline', 
                                        xmin = xmin, 
                                        xmax = xmax)
            cov_file_ad = os.path.join(idp_visit_dir, 'cov_bspline_ad.txt')          
            np.savetxt(cov_file_ad, X_ad)
            
            # save the responses for the adaptation data
            resp_file_ad = os.path.join(idp_visit_dir, 'resp_ad.txt') 
            y_ad = df_ad[idp].to_numpy()
            np.savetxt(resp_file_ad, y_ad)
        
            # save the site ids for the adaptation data
            sitenum_file_ad = os.path.join(idp_visit_dir, 'sitenum_ad.txt') 
            site_num_ad = df_ad['sitenum'].to_numpy(dtype=int)
            np.savetxt(sitenum_file_ad, site_num_ad)
            
            # save the site ids for the test data 
            sitenum_file_te = os.path.join(idp_visit_dir, 'sitenum_te.txt')
            site_num_te = df_te['sitenum'].to_numpy(dtype=int)
            np.savetxt(sitenum_file_te, site_num_te)

            # adaptation files are among inputs to adjust the offset 
            y, yhat_te, s2_te, Z = predict(cov_file_te, 
                                        alg = 'blr', 
                                        respfile = resp_file_te, 
                                        model_path = os.path.join(idp_dir,'Models'),
                                        adaptrespfile = resp_file_ad,
                                        adaptcovfile = cov_file_ad,
                                        adaptvargroupfile = sitenum_file_ad,
                                        testvargroupfile = sitenum_file_te)
