import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from enigmatoolbox.utils.parcellation import parcel_to_surface
from enigmatoolbox.plotting import plot_cortical, plot_subcortical
from enigmatoolbox.utils.useful import reorder_sctx

def load_clinics(clinics):
    # rename the czech columns
    clinics.columns = ["Visit_ID", "HYDRA_ID", "Full_HYDRA_ID", "Special_ID", "Inclusion", 
                        "Inclusion_Comment", "Category", "Birth", "sex", "Comment",
                        'age', 'Date_of_Visit', 'No_Visit', "Dir_Name", 'Visit_Comment','Project',
                        'Completion_Comment', 'List_of_Series','site', 'Siblings_Patients',
                        'Siblings_Healthy', 'Laterality_EHI', 'Right_Handed']

    # Change the values of categorical variables from czech to english
    clinics['Category'] = clinics['Category'].str[:].str.upper().map({'PACIENT':'Patient', 'KONTROLA':'Control', 'SOUROZENEC':'Sibling', 'HIGH RISK':'High_risk'})
    clinics['Inclusion'] = clinics['Inclusion'].str[:].str.upper().map({'ZAŘAZENA':'Included', 'ZAŘAZENA S VÝHRADAMI':'Included with Reservations'})

    # we need to create a correct index for merging dataframes
    idx = clinics["Dir_Name"]

    for iid in range(0, clinics.shape[0]):
        iname = idx[iid]
        if isinstance(iname, str):
            name = iname.split("_")[1]+ '_' + iname.split("_")[-1]
        else:
            name = 'no'
        
        if iid == 0:
            names = name
        else:
            names = np.append(names,name)


    clinics.index = names

    # drop unnecessary variables
    clinics = clinics[["Category","Sex", "Birth", 'Age_at_Visit', "Inclusion", "No_Visit", "Right_Handed"]]

    # drop Siblings and High Risk
    clinics = clinics[(clinics["Category"] == "Patient") |(clinics["Category"] == "Control")]

    if 'no' in clinics.index:
        clinics = clinics.drop(index=['no'])
    
    # wrap up
    return(clinics)
    


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
    # return(cov_norm, cov_pat, sfeat_norm, sfeat_test) = prepare_data(data_final, transformation, nm_dir, **kwargs)
    
    features = kwargs.get('features', False)

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
            idc = [s for s in list(data_final.columns) if 'nvox_' in s]
        else:
            print(('Something is wrong with:' + features + 'Are you using a wrong feature name?'))

        feat_norm = data_final[idc].loc[data_final['Category']=='Control']
        feat_test = data_final[idc].loc[data_final['Category']=='Patient']
        df_idc = pd.DataFrame(idc)
        df_idc.to_csv(os.path.join(nm_dir,'colnames.txt'), sep=' ', header=False, index=False)
    
    else:
        ## Features - Desikan Killiany Atlas
        # indices of DK
        i_start = list(data_final.columns).index("rh_bankssts")
        i_end = list(data_final.columns).index("lh_insula")+1
        df_idc = pd.DataFrame(data_final.columns[range(i_start,i_end)].to_numpy())
        df_idc.to_csv(os.path.join(nm_dir,'colnames.txt'), sep=' ', header=False, index=False)
        feat_norm = data_final.loc[data_final['Category']=='Control', data_final.columns[range(i_start,i_end)]]
        feat_test = data_final.loc[data_final['Category']=='Patient', data_final.columns[range(i_start,i_end)]]
        
    ## Apply transformation ['no', 'zscore', 'scale']
    if transformation == 'no':
        print('No transformation was applied')
        sfeat_norm = feat_norm
        sfeat_test = feat_test

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

    return(cov_norm, cov_pat, sfeat_norm, sfeat_test)


def plot_quality(nm_dir, **kwargs):
    
    show_img = kwargs.get('show_img', True)
    save_img = kwargs.get('save_img', False)
    fs_var =  kwargs.get('fs_var', None)
    images_dir = kwargs.get('images_dir', None)
    transformation =  kwargs.get('transformation', None)

    ## Load and plot the quality measures
    # Standardized Mean Squared Error (closer 0 = better)
    SMSE = pd.read_csv(os.path.join(nm_dir, 'SMSE_estimate.txt'), header=None)
    # Explained Variance (closer 1 = betetr)
    EXPV = pd.read_csv(os.path.join(nm_dir, 'EXPV_estimate.txt'), header=None)
    # Mean Standardized Log Loss (the negative, the better)
    MSLL = pd.read_csv(os.path.join(nm_dir, 'MSLL_estimate.txt'), header=None)
    # Correlation
    Rho = pd.read_csv(os.path.join(nm_dir, 'Rho_estimate.txt'), header=None)
    pRho = pd.read_csv(os.path.join(nm_dir, 'pRho_estimate.txt'), header=None)

    # Put everything into dataframe
    measures = ['SMSE', 'EXPV', 'MSLL', 'Rho', 'pRho']
    nmeasures = pd.concat([SMSE,EXPV,MSLL,Rho,pRho], axis=1)
    nmeasures.columns = measures

    # Plotting
    if show_img:
        sns.set_theme(style = 'white')
        fig, axes = plt.subplots(2,3, sharex=False, figsize=(10,7))

        j=0
        for i, imeasure in enumerate(measures):
            
            i=np.divmod(i,3)[1]

            g = sns.histplot(data=nmeasures[imeasure], ax=axes[j,i])
            g.set_title(imeasure)
            
            if i==2:
                j=1

        fig.tight_layout()
        sns.despine()
        plt.show()
        
        if save_img:
            fig.savefig(os.path.join(images_dir, ('03_hist' + fs_var + '_'+transformation+'.png')))
    
    # wrap up
    return(nmeasures)


# confidence interval calculation at x_forward
def myround(x, base=10):
    return ([base * round(i/base) for i in x ])

def confidence_interval(s2,x,x_forward,z):
  CI=np.zeros((len(x_forward),s2.shape[1]))
  for i,xdot in enumerate(x_forward):
    ci_inx=np.isin(myround(x),xdot)
    S2=s2[ci_inx]
    S_hat=np.mean(S2,axis=0)
    n=S2.shape[0]
    CI[i,:]=z*np.power(S_hat/n,.5)
  return CI

# trajectory plotting
def trajectory_plotting(cov_forw, nm_dir):

    # load forward predictions
    forward_yhat = pd.read_csv(os.path.join(nm_dir,'yhat_forward.txt'), sep = ' ', header=None)
    feat_norm = pd.read_csv(os.path.join(nm_dir,'feat_norm.txt'), sep = ' ',header=None)
    cov_norm = pd.read_csv(os.path.join(nm_dir,'cov_norm.txt'), sep = ' ',header=None)
    feature_names = pd.read_csv(os.path.join(nm_dir,'colnames.txt'), sep = ' ',header=None).to_numpy()
    feature_names = [i[0] for i in feature_names]
    sex_covariates=['Female','Male']
    os.chdir(nm_dir)

    for i, sex in enumerate(sex_covariates):

        yhat_forward=forward_yhat.values
        y_params = int(cov_forw.shape[0]/2)
        yhat_forward=yhat_forward[y_params*i:y_params*(i+1)]
        x_forward=list(cov_forw.iloc[range(0,y_params),0])

        # Find the index of the data exclusively for one sex. Female: 0; Male: 1;
        idx=np.where(cov_norm[1]==i)[0]
        x=cov_norm.values[idx,0]

        # read data, filter features
        y = pd.read_csv(os.path.join(nm_dir,'feat_norm.txt'), sep = ' ', header=None)
        y = y.values[idx]

        # confidence Interval yhat+ z *(std/n^.5)-->.95 % CI:z=1.96, 99% CI:z=2.58
        s2= pd.read_csv(os.path.join(nm_dir, 'ys2_estimate.txt'), sep = ' ', header=None)
        s2=s2.values[idx]

        CI_95=confidence_interval(s2,x,x_forward,1.96)
        CI_99=confidence_interval(s2,x,x_forward,2.58)

        CI_95[CI_95 > 1e+04] = np.nan
        CI_99[CI_99 > 1e+04] = np.nan

        # Create a trajectroy for each point    
        for j,name in enumerate(feature_names[0:1]):
            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.plot(x_forward,yhat_forward[:,j], linewidth=4, label='Normative trejactory')


            ax.plot(x_forward,CI_95[:,j]+yhat_forward[:,j], linewidth=2,linestyle='--',c='g', label='95% confidence interval')
            ax.plot(x_forward,-CI_95[:,j]+yhat_forward[:,j], linewidth=2,linestyle='--',c='g')

            ax.plot(x_forward,CI_99[:,j]+yhat_forward[:,j], linewidth=1,linestyle='--',c='k', label='99% confidence interval')
            ax.plot(x_forward,-CI_99[:,j]+yhat_forward[:,j], linewidth=1,linestyle='--',c='k')

            ax.scatter(x,y[:,j],c='r', label=name)
            plt.legend(loc='upper left')
            plt.title('Normative trejectory of' +name+' in '+ sex +' cohort')
            plt.show()
            plt.close()



def dk_roi_viz(nm_dir, z_test, thresh, vis, fs_var):
    # 3d visualization of df atlas (grey and wm)
    # dk_roi_viz(nm_dir, z_test, thresh, vis, **kwargs):

    # for WM
    if 'IC' in fs_var:
        cnames = pd.read_csv(os.path.join(nm_dir,'colnames.txt'), sep = ' ',header=None).to_numpy()
        cnames = [i[0] for i in cnames]
        cnames = [i.split('_')[1] for i in cnames]
        z_test.columns = cnames

        # This is for vizualization of Desikan Killiany intracranial volumes
        int_dict = {'Left-Lateral-Ventricle': 'LLateVent',
                'Left-Thalamus': 'Lthal',
                'Left-Caudate': 'Lcaud',
                'Left-Putamen': 'Lput',
                'Left-Pallidum': 'Lpal',
                'Left-Hippocampus': 'Lhippo',
                'Left-Amygdala': 'Lamyg',
                'Left-Accumbens-area': 'Laccumb',
                'Right-Lateral-Ventricle': 'RLatVent',
                'Right-Thalamus': 'Rthal',
                'Right-Caudate': 'Rcaud',
                'Right-Putamen': 'Rput',
                'Right-Pallidum': 'Rpal',
                'Right-Hippocampus': 'Rhippo',
                'Right-Amygdala': 'Ramyg',
                'Right-Accumbens-area': 'Raccumb'
        }

        z_test.rename(columns=int_dict, inplace=True)
        z_pk = z_test.iloc[:,[0,4,5,6,7,11,12,14,18,22,23,24,25,26,27,28]]
        z_reorder = reorder_sctx(z_pk)

        if vis == 'pos':
            z_epoz = (z_reorder>thresh).sum()/z_test.shape[0]
            return(z_epoz)
        elif vis == 'neg':
            z_eneg = (z_reorder<-thresh).sum()/z_test.shape[0]
            return(z_eneg)
        else:
            print('You can only pick poz or neg type of visualization')
    
    # for GM
    else:
        # prepare data for visualization
        cnames = pd.read_csv(os.path.join(nm_dir,'colnames.txt'), sep = ' ',header=None).to_numpy()
        cnames = [i[0] for i in cnames]
        cnames = [i.replace('rh_','R_') for i in cnames]
        cnames = [i.replace('lh_','L_') for i in cnames]
        z_test.columns = cnames

        R_hemi = [col for col in z_test.columns if 'R_' in col]
        L_hemi = [col for col in z_test.columns if 'L_' in col]
        z_test = pd.concat([z_test[L_hemi], z_test[R_hemi]], axis=1)


        if vis == 'pos':
            # plot the ratio of extreme positive deviations
            z_epoz = (z_test>thresh).sum()/z_test.shape[0]
            z_epoz_parc = parcel_to_surface(z_epoz, 'aparc_fsa5')
            return(z_epoz_parc)

            
        elif vis == 'neg':
            # plot the ratio of extreme negative deviations
            z_eneg = (z_test<-thresh).sum()/z_test.shape[0]
            z_epoz_neg = parcel_to_surface(z_eneg, 'aparc_fsa5')
            return(z_epoz_neg)
            
        else:
            print('You can only pick poz or neg type of visualization')