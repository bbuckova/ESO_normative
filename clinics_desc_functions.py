import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

def merge_and_reorder(clinics, neuro):
    """
    Function that creates various datasets based on the freesurfer input
    Additionaly, it reorders columns, so that symetrical brain ROIs are next to each other
    merge_and_reorder(clinics, neuro)
    clinics = pandas dataframe with clinical information
    neuro = pandas dataframe with freesurfer information     
    """
    # Merge the two dataframes
    clinics_to_merge = clinics
    data = pd.concat([clinics_to_merge, neuro], axis=1, join="inner")

    # Reordering columns here, so that we can compare the right and left hemisphere better
    cols = data.columns.to_list()
    ocols = [cols[i] for i in range(0,clinics_to_merge.shape[1])]
    for i in range(0,int(neuro.shape[1]/2)):
        ocols.append(cols[i + clinics_to_merge.shape[1]])
        ocols.append(cols[i + clinics_to_merge.shape[1] + int(neuro.shape[1]/2)])

    data = data[ocols]
    
    # wrap up
    return(data)


def desc_boxplots(data, fs_suffix, images_dir, **kwargs):
    # Plots boxplots of freesurfer ROIs across categories (Control, Patient, Sibling, High risk)
    # The images are saved
    # desc_boxplots(data, fs_suffix, images_dir)
    
    save_img = kwargs.get('save_img', True)

    matplotlib.rcParams['figure.figsize'] = [15.7, 8.27]
    matplotlib.rcParams['savefig.dpi'] = 300
    matplotlib.rcParams["figure.titlesize"] = "xx-large"

    # The initial boxplot
    sns.set_style("whitegrid")
    sns.color_palette("Spectral", as_cmap=True)

    # get indexes of fsvars
    i_start = list(data.columns).index("rh_bankssts")
    i_end = list(data.columns).index("lh_insula")+1

    # Set axes - to be comparable
    how_much_to_round = (len(str(data.iloc[:,range(i_start,i_end)].max().max()).split('.')[0])-1)*-1
    ymax = round(data.iloc[:,range(i_start,i_end)].max().max(), how_much_to_round)

    # Plot boxplots across categories
    nplots = len(data["Category"].unique())
    fig, axes = plt.subplots(nplots,1, sharex=True,figsize=(15,nplots*3))
    fig.suptitle('Visit 1:' +fs_suffix)

    for ipic, cat in enumerate(data["Category"].unique()):
        axes[ipic].set_title(cat+' (n = ' + str(data["Category"].value_counts()[ipic])+')')
        g = sns.boxplot(ax = axes[ipic], data=data[(data["Category"] == cat)].iloc[:,range(i_start,i_end)], palette="Spectral" )
        g.set_xticklabels(g.get_xticklabels(),rotation = 90)
        g.set(ylim=(0,ymax))
        sns.despine()

    # save image
    if save_img:
        fig.savefig(os.path.join(images_dir, ('01_bp_' + fs_suffix + '.png')))   

def en_qc(data, **kwargs):
    ## QC - based on the euler number
    # output from freesurfer = no.holes (computation based on the euler number)
    # EN = 2-2*no.holes
    #data_final = en_qc(data)
    
    save_img = kwargs.get('save_img', False)
    images_dir = kwargs.get('img_dir', None)
    show_img = kwargs.get('show_img', False)

    indices = [i for i, s in enumerate(list(data.columns)) if 'Holes' in s]
    en_mean = (data.iloc[:,indices[0]]*-2+2 + data.iloc[:,indices[1]]*-2+2)/2
    site_median = en_mean.median()
    en_final = np.sqrt(np.absolute((en_mean-site_median)*(-1)))

    # Plotting estimated TIV across Patients/Controls and Males/Females
    if show_img == True:
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, sharex=True,figsize=(7,3))
        matplotlib.rcParams["figure.titlesize"] = "large"
        fig.suptitle('Euler numbers (sqrt(abs(centered)))')
        g = sns.boxplot(data=en_final, orient='h', linewidth=2.5)
        sns.despine()

        if save_img == True:
            fig.savefig(os.path.join(images_dir, ('en_qc.png')))   

    # remove the outliers
    id = en_final>10
    id_to_remove = id[id].index
    data_final = data.drop(axis=0, index=id_to_remove)
    return(data_final)


def prepare_data(data_final, transformation, nm_dir, **kwargs):

    features = kwargs.get('features', False)

    # Create the directory with results
    nm_dir = os.path.join(main_dir, fs_var+'_'+transformation)
    os.makedirs(nm_dir, exist_ok=True)

    ## Covariates - Age and Sex
    # Controls
    cov_norm = data_final.loc[data_final['Category']=='Control', ["Age_at_Visit","Sex"]]
    cov_norm.to_csv(os.path.join(nm_dir,'cov_norm.txt'), sep=' ', header= False, index=False)
    
    # Patients
    cov_pat = data_final.loc[data_final['Category']=='Patient', ["Age_at_Visit","Sex"]]
    cov_pat.to_csv(os.path.join(nm_dir,'cov_test.txt'), sep=' ', header= False, index=False)

    ## Features 'IC based features'
    if 'IC' in features:

        if 'Mean' in features:
            idc = [s for s in list(data_final.columns) if 'mean_' in s]
        elif 'Std' in features:
            idc = [s for s in list(data_final.columns) if 'std_' in s]
        elif 'Vol' in features:
            idc = [s for s in list(data_final.columns) if 'novx_' in s]
        else:
            print(('Something is wrong with:' + features + 'Are you using a wrong feature name?'))

        feat_norm = data_final.loc[data_final['Category']=='Control', data_final.columns[idc]]
        feat_test = data_final.loc[data_final['Category']=='Patient', data_final.columns[idc]]
        idc.to_csv(os.path.join(nm_dir,'colnames.txt'), sep=' ', header=False, index=False)
    
    else:
        ## Features - Desikan Killiany Atlas
        # indices of DK
        i_start = list(data_final.columns).index("rh_bankssts")
        i_end = list(data_final.columns).index("lh_insula")+1
        feat_norm = data_final.loc[data_final['Category']=='Control', data_final.columns[range(i_start,i_end)]]
        feat_test = data_final.loc[data_final['Category']=='Patient', data_final.columns[range(i_start,i_end)]]
        
    ## Apply transformation ['no', 'zscore', 'scale']
    if transformation == 'no':
        print('No transformation was applied')
        feat_norm.to_csv(os.path.join(nm_dir,'feat_norm.txt'), sep=' ', header=False, index=False)
        feat_test.to_csv(os.path.join(nm_dir,'feat_test.txt'), sep=' ', header=False, index=False)
    
    elif transformation == 'zscore':
        print('Z-transformation was applied')
        sfeat_norm = (feat_norm-feat_norm.mean())/feat_norm.std()
        sfeat_test = (feat_test-feat_norm.mean())/feat_norm.std()
        
        sfeat_norm.to_csv(os.path.join(nm_dir,'feat_norm.txt'), sep=' ', header=False, index=False)
        sfeat_test.to_csv(os.path.join(nm_dir,'feat_test.txt'), sep=' ', header=False, index=False)
    
    elif transformation == 'scale': # use this for mm^3
        print('Scaling applied')
        sfeat_norm = feat_norm/1000
        sfeat_test = feat_test/1000
        
        sfeat_norm.to_csv(os.path.join(nm_dir,'feat_norm.txt'), sep=' ', header=False, index=False)
        sfeat_test.to_csv(os.path.join(nm_dir,'feat_test.txt'), sep=' ', header=False, index=False)
        
    else:
        print("Wrong transformation")

