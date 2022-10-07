# Functions neede for the simulation study (will be later transfered to custom --- after debugging)
# Function to generate Subjects
def generate_longitudinal_subjects(model_path, simulations_dir, no_females = 1000, var_population = 1, var_noise = 0.5, effect='none', effect_size = 0, effect_var = 0):
    """
    Function will generate brain data based on coeffitiens from chosen model
    generate_longitudinal_subjects(model_path, simulations_dir, no_females = 1000, var_population = 1, var_noise = 0.5):
    
    model_path... path of model with coeffitients
    simulations_dir... where the results are to be stored
    no_females... number of females (dataset is going to be twice the size, default = 1000)
    var_population... variation in population (default = 1)
    var_noise... variation in noise (default = 0.5)
    effect...   - none (use to generate controls)
                - uniform
                - normal
    effect_size... effect size (default = 0)
    effect_var... effect variation (default = 0)
    """
    from pcntoolkit.normative import estimate, predict, evaluate

    # Create template dataset for V1 and V2
    #elems = np.arange(18,60,1)
    #v1_age = np.repeat(elems,no_females/100)
    #v1_age = np.concatenate([v1_age,v1_age])
    v1_age = np.random.randint(18,60,size=no_females*2)
    

    v2_age = v1_age+1

    sex = np.concatenate([np.zeros([no_females]),np.ones([no_females])])
    id = ['c'+str(i) for i in range(1,no_females*2+1)]

    v1_template = pd.DataFrame(np.array([v1_age,sex]).T, columns=['age', 'sex'], index=id)
    v2_template = pd.DataFrame(np.array([v2_age,sex]).T, columns=['age', 'sex'], index=id)

    v1_template['site'] = 'simulation'
    v1_template['sitenum'] = 4223
    v2_template['site'] = 'simulation'
    v2_template['sitenum'] = 4223

    # Deterministic part of the simulation
    # phi(x)*w --> Yhat
    cols_cov = ['age','sex']
    v1_covars = create_design_matrix(v1_template[cols_cov], 
                                site_ids = v1_template['site'],
                                all_sites = site_ids_tr,
                                basis = 'bspline', 
                                xmin = -5, 
                                xmax = 110)

    v2_covars = create_design_matrix(v2_template[cols_cov], 
                                site_ids = v2_template['site'],
                                all_sites = site_ids_tr,
                                basis = 'bspline', 
                                xmin = -5, 
                                xmax = 110)


    v1_fsaveto = os.path.join(simulations_dir,'v1_controls.txt')
    np.savetxt(v1_fsaveto, v1_covars)
    v2_fsaveto = os.path.join(simulations_dir,'v2_controls.txt')
    np.savetxt(v2_fsaveto, v2_covars)

    from temp_func_pcn import predict
    v1_template['Yhat'], v1_s2_orig = predict(v1_fsaveto, respfile = None, alg='blr', model_path=model_path)
    v2_template['Yhat'], v2_s2_orig = predict(v2_fsaveto, respfile = None, alg='blr', model_path=model_path)

    # Adding the position within the population to every individual
    np.random.seed(42)

    if var_population == 'model':
        with open(os.path.join(model_path, 'NM_0_0_estimate.pkl'), 'rb') as handle:
            nm = pickle.load(handle) 
        beta, junk1, junk2 = nm.blr._parse_hyps(nm.blr.hyp, v1_covars)        
        v1_template['pop'] = np.random.normal(0, 1/beta, size=v1_template.shape[0])
        beta = 1/beta
    else:
        v1_template['pop'] = np.random.normal(0,var_population,size=v1_template.shape[0])
        beta = var_population + var_noise

    # Adding the effect if there is any (controls vs patients)
    if effect == 'none':
        effect_size = 0
    elif effect == 'normal':
        effect_size = np.random.normal(effect_size, effect_var, size=no_patients)
    elif effect == 'uniform':
        effect_size = effect_size

    v1_template['effect_size'] = effect_size
    v1_template['Yhat_pop'] = v1_template['Yhat'] + v1_template['pop'] + v1_template['effect_size']
    v1_template['noise'] = np.random.normal(0,var_noise,size=v1_template.shape[0])
    v1_template['warped'] = v1_template['Yhat'] + v1_template['pop'] + v1_template['effect_size'] + v1_template['noise']

    v2_template['pop'] = v1_template['pop']
    v2_template['effect_size'] = effect_size
    v2_template['Yhat_pop'] = v2_template['Yhat'] + v2_template['pop'] + v2_template['effect_size']
    v2_template['noise'] = np.random.normal(0,var_noise,size=v2_template.shape[0])
    v2_template['warped'] = v2_template['Yhat'] + v2_template['pop'] + v2_template['effect_size'] + v2_template['noise']


    # Now warping into the original space
    # load the normative model
    with open(os.path.join(model_path, 'NM_0_0_estimate.pkl'), 'rb') as handle:
        nm = pickle.load(handle) 

    # get the warp and warp parameters
    W = nm.blr.warp
    warp_param = nm.blr.hyp[1:nm.blr.warp.get_n_params()+1] 
        
    # first, we warp predictions for the true data and compute evaluation metrics
    v1_template[random_idp] = W.warp_predictions(np.squeeze(v1_template['warped'].to_numpy()), np.squeeze(beta), warp_param)[0]
    v2_template[random_idp] = W.warp_predictions(np.squeeze(v2_template['warped'].to_numpy()), np.squeeze(beta), warp_param)[0]

    return(v1_template, v2_template, beta)


###
# Train_test_split - split across sites, keep 30% as TESTING
###
def run_and_attach(v1_cont, v2_cont, v1_pat, v2_pat, simulations_dir, random_idp):

    from temp_func import pretrained_adapt_controls

    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.25, random_state=42)
    
    for train, test in sss.split(v1_cont.index,  v1_cont['sex']):

        #iter_dir = isim_dir
        #os.makedirs(iter_dir, exist_ok=True)
        site_ids_te =  sorted(set(v1_pat['site'].to_list()))

        for ivisit in range(1,3):
            if ivisit == 1:
                df_te = v1_pat
                df_tec = v1_cont.iloc[test]
                df_ad = v1_cont.iloc[train]

            elif ivisit == 2:
                df_te = v2_pat
                df_tec = v2_cont.iloc[test]
                df_ad = v1_cont.iloc[train]

            visit_dir = os.path.join(simulations_dir, 'V'+str(ivisit))
            os.makedirs(visit_dir, exist_ok=True)
            

            #custom.pretrained_adapt_small(random_idp, site_ids_tr, site_ids_te, model_path, visit_dir, df_ad, df_te)
            pretrained_adapt_controls([random_idp], site_ids_tr, site_ids_te, models_pretrained_all, visit_dir, df_ad, df_tec, df_te)

            # create textfiles for checking
            df_te.to_csv(os.path.join(visit_dir,random_idp, 'patients.csv'), sep=' ', index=True)
            df_tec.to_csv(os.path.join(visit_dir,random_idp, 'controls_test.csv'), sep=' ', index=True)
            df_ad.to_csv(os.path.join(visit_dir, random_idp, 'controls_adapt.csv'), sep=' ', index=True)

    # Attach
    # Load testing controls and poatients (original dataframes) and attach the z-scores
    v1_cont = pd.read_table(os.path.join(simulations_dir, 'V1', random_idp, 'controls_test.csv'), sep=' ', index_col=0)
    v2_cont = pd.read_table(os.path.join(simulations_dir, 'V2', random_idp, 'controls_test.csv'), sep=' ', index_col=0)

    v1_pat = pd.read_table(os.path.join(simulations_dir, 'V1', random_idp, 'patients.csv'), sep=' ', index_col=0)
    v2_pat = pd.read_table(os.path.join(simulations_dir, 'V2', random_idp, 'patients.csv'), sep=' ', index_col=0)

    v1_cont_z = pd.read_table(os.path.join(simulations_dir, 'V1', random_idp, 'Z_conttest.txt'), sep=' ', header=None)
    v2_cont_z = pd.read_table(os.path.join(simulations_dir, 'V2', random_idp, 'Z_conttest.txt'), sep=' ', header=None)

    v1_pat_z = pd.read_table(os.path.join(simulations_dir, 'V1', random_idp, 'Z_predict.txt'), sep=' ', header=None)
    v2_pat_z = pd.read_table(os.path.join(simulations_dir, 'V2', random_idp, 'Z_predict.txt'), sep=' ', header=None)

    v1_cont_yhat = pd.read_table(os.path.join(simulations_dir, 'V1', random_idp, 'yhat_conttest.txt'), sep=' ', header=None)
    v2_cont_yhat = pd.read_table(os.path.join(simulations_dir, 'V2', random_idp, 'yhat_conttest.txt'), sep=' ', header=None)

    v1_pat_yhat = pd.read_table(os.path.join(simulations_dir, 'V1', random_idp, 'yhat_predict.txt'), sep=' ', header=None)
    v2_pat_yhat = pd.read_table(os.path.join(simulations_dir, 'V2', random_idp, 'yhat_predict.txt'), sep=' ', header=None)

    v1_cont_y = pd.read_table(os.path.join(simulations_dir, 'V1', random_idp, 'y_conttest.txt'), sep=' ', header=None)
    v2_cont_y = pd.read_table(os.path.join(simulations_dir, 'V2', random_idp, 'y_conttest.txt'), sep=' ', header=None)

    v1_pat_y = pd.read_table(os.path.join(simulations_dir, 'V1', random_idp, 'y_predict.txt'), sep=' ', header=None)
    v2_pat_y = pd.read_table(os.path.join(simulations_dir, 'V2', random_idp, 'y_predict.txt'), sep=' ', header=None)

    # Attach the z-scores to the original dataframes
    v1_cont[random_idp+'_z_orig'] = v1_cont_z[0].to_numpy()
    v1_cont[random_idp+'_yhat'] = v1_cont_yhat[0].to_numpy()
    v1_cont[random_idp+'_y'] = v1_cont_y[0].to_numpy()

    v2_cont[random_idp+'_z_orig'] = v2_cont_z[0].to_numpy()
    v2_cont[random_idp+'_yhat'] = v2_cont_yhat[0].to_numpy()
    v2_cont[random_idp+'_y'] = v2_cont_y[0].to_numpy()

    v1_pat[random_idp+'_z_orig'] = v1_pat_z[0].to_numpy()
    v1_pat[random_idp+'_yhat'] = v1_pat_yhat[0].to_numpy()
    v1_pat[random_idp+'_y'] = v1_pat_y[0].to_numpy()

    v2_pat[random_idp+'_z_orig'] = v2_pat_z[0].to_numpy()
    v2_pat[random_idp+'_yhat'] = v2_pat_yhat[0].to_numpy()
    v2_pat[random_idp+'_y'] = v2_pat_y[0].to_numpy()

    return v1_cont, v2_cont, v1_pat, v2_pat

# LONGITUDINAL PLOTTING FUNCTION
# check the controls are not dramatically different between the two visits
# x coordinates - age - two rows of data
def plot_longitudinal(v1_age, v2_age,  v1_Yhat, v2_Yhat, to_plot = True, v1_col='lightblue', v2_col='lightcoral', tick_col = 'gray', title = '', xlabel = 'Age', ylabel = 'Yhat'):
    """
    The function will plot the longitudinal data as connected dots
    plot_longitudinal(v1_age, v2_age,  v1_Yhat, v2_Yhat, to_plot = True, v1_col='lightblue', v2_col='lightcoral', tick_col = 'gray', title = '', xlabel = 'Age', ylabel = 'Yhat')
    
    v1_age, v2_age... what will be plotted on x axis, ideally 1xn 2D array
    v1_Yhat, v2_Yhat... what will be plotted on y axis, ideally 1xn 2D array
    to_plot... if True, the plot will be shown, if False, the plot will be returned
    v1_col, v2_col, tick_col... color of the dots for the first and second visit
    title, xlabel, ylabel... title and labels of the plot
    """
    def check_dimensions(vector):
        """
        check_dimensions(vector)
        reshapes array into 2D 1xn array
        """
        if len(vector.shape) == 1:
            return vector[np.newaxis,:]
        elif len(vector.shape) == 2:
            if vector.shape[0] > vector.shape[1]:
                return vector.T
            else:
                return vector
        else:
            return False

    v1_age = check_dimensions(v1_age)
    v2_age = check_dimensions(v2_age)
    v1_Yhat = check_dimensions(v1_Yhat)
    v2_Yhat = check_dimensions(v2_Yhat)
    
    x_coords = np.concatenate([v1_age, v2_age],axis=0)
    y_coords = np.concatenate([v1_Yhat, v2_Yhat],axis=0)

    if to_plot:
        fig, ax = plt.subplots()
        plt.plot(x_coords, y_coords, color=tick_col)
        plt.scatter(v1_age, v1_Yhat, color=v1_col, s=20, alpha = 0.5)
        plt.scatter(v2_age, v2_Yhat, color=v2_col, s=20, alpha = 0.5)

    return ax




# Loading packages
import os, glob, pickle, time, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pcntoolkit.util.utils import create_design_matrix
import xarray as xr

# custom functions
code_dir = ('/home/barbora/Documents/Projects/Normative_Models/ESO/code')
os.chdir(code_dir)
import clinics_desc_functions as custom

custom.set_seed(seed=42)

# Paths to directoris
project_dir = ('/home/barbora/Documents/Projects/Normative_Models')
simulations_dir = '/home/barbora/Documents/Projects/Normative_Models/ESO/simulations'
models_dir = '/home/barbora/Documents/Projects/Normative_Models/ESO/braincharts/models/lifespan_57K_82sites/'
models_pretrained_all = '/home/barbora/Documents/Projects/Normative_Models/ESO/braincharts/'
paper_im_dir = os.path.join(project_dir,'ESO', 'draft', 'img')

# pretrained model
model_name, site_names, site_ids_tr, idp_ids = custom.pretrained_ini()
thick_idp = [i for i in idp_ids if 'thick' in i]

# create empty data array (3D) to store the results in
data = np.empty([len(np.arange(0.5, 5.5, 0.5)) * len(np.arange(0.25, 2.25, 0.25)), 6, len(thick_idp)])

###
# This is going to be the first cycle acroos idps
###
for i, random_idp in enumerate(thick_idp):#np.random.randint(len(idp_ids), size=1)
    print(random_idp)

    # one random model - just trying to get some reasonable coeffitients
    model_path = os.path.join(models_dir,random_idp,'Models')

    # to define row in the data
    irow = 0
    for ivar_population in np.arange(0.5, 5.5, 0.5):
        print(str(ivar_population))
        for ivar_noise in np.arange(0.25, 2, 0.25):
            ###
            # Generate controls and patients
            ###
            v1_cont, v2_cont, s2_model = generate_longitudinal_subjects(model_path, simulations_dir, no_females = 1000, var_population = ivar_population, var_noise = ivar_noise, effect='none')
            v1_pat, v2_pat, s2_model = generate_longitudinal_subjects(model_path, simulations_dir, no_females = 100, var_population = ivar_population, var_noise = ivar_noise, effect='none')
            
            ###
            # potential effect - not now, only checking controls
            #v2_pat[random_idp] = v2_pat[random_idp] - 0.2

            ###
            # Run and attach the results of normative model
            v1_cont, v2_cont, v1_pat, v2_pat = run_and_attach(v1_cont, v2_cont, v1_pat, v2_pat, simulations_dir, random_idp)

            ###
            # Statistics
            nom = ((v2_cont[random_idp+'_y'] - v2_cont[random_idp+'_yhat']) - (v1_cont[random_idp+'_y'] - v1_cont[random_idp+'_yhat']))
            cont_var = ((v2_cont[random_idp+'_y'] - v2_cont[random_idp+'_yhat']) - (v1_cont[random_idp+'_y'] - v1_cont[random_idp+'_yhat'])).var()
            cont_z = nom/np.sqrt(cont_var)

            nom = ((v2_pat[random_idp+'_y'] - v2_pat[random_idp+'_yhat']) - (v1_pat[random_idp+'_y'] - v1_pat[random_idp+'_yhat']))
            pat_z = nom/np.sqrt(cont_var)

            data[irow, 0, i] = ivar_population
            data[irow, 1, i] = ivar_noise
            data[irow, 2, i] = cont_z.mean()
            data[irow, 3, i] = cont_z.var()
            data[irow, 4, i] = pat_z.mean()
            data[irow, 5, i] = pat_z.var()

            irow += 1


df_data = xr.DataArray(data, 
                        dims=['iters', 'stats', 'idp'], 
                        coords={'iters': range(0,80),
                                'stats': ['pop_var', 'noise_var', 'cont_mean', 'cont_var', 'pat_mean', 'pat)var'], 
                                'idp': thick_idp[0:2]},
                        name = 'features'
                        )
#df_data = xr.DataArray(data, dims=['iters', 'stats', 'idp'], coords={'iters': [1,2,3], 'stats': ['mean','var'], 'idp': thick_idp})
df_data.to_netcdf(os.path.join(simulations_dir, "simulations_stats.nc"))
