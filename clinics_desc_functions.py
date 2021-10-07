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
    clinics_to_merge = clinics[["Category","Sex", "Birth", 'Age_at_Visit', "Inclusion", "No_Visit", "Right_Handed"]]
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


def desc_boxplots(data, fs_suffix, images_dir):
    # Plots boxplots of freesurfer ROIs across categories (Control, Patient, Sibling, High risk)
    # The images are saved
    # desc_boxplots(data, fs_suffix, images_dir)
    
    matplotlib.rcParams['figure.figsize'] = [15.7, 8.27]
    matplotlib.rcParams['savefig.dpi'] = 300
    matplotlib.rcParams["figure.titlesize"] = "xx-large"

    # The initial boxplot
    sns.set_style("whitegrid")
    sns.color_palette("Spectral", as_cmap=True)

    # Set axes - to be comparable
    how_much_to_round = (len(str(data.iloc[:,range(7,data.shape[1])].max().max()).split('.')[0])-1)*-1
    ymax = round(data.iloc[:,range(7,data.shape[1])].max().max(), how_much_to_round)

    # Plot boxplots across categories
    fig, axes = plt.subplots(4,1, sharex=True,figsize=(15,12))
    fig.suptitle('Visit 1:' +fs_suffix)

    for ipic, cat in enumerate(data["Category"].unique()):
        axes[ipic].set_title(cat+' (n = ' + str(data["Category"].value_counts()[ipic])+')')
        g = sns.boxplot(ax = axes[ipic], data=data[(data["Category"] == cat)].iloc[:,range(7,data.shape[1])], palette="Spectral", )
        g.set_xticklabels(g.get_xticklabels(),rotation = 90)
        g.set(ylim=(0,ymax))
        sns.despine()

    # save image
    fig.savefig(os.path.join(images_dir, ('01_bp_' + fs_suffix + '.png')))   
