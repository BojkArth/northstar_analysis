#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib as mpl
import json
import os
import errno
from sklearn.manifold import TSNE

import northstar
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

import sys
sys.path.append('/home/bojk/Data/minimeta_pyfiles/')

mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['figure.titlesize'] = 30
mpl.rcParams['legend.fontsize'] = 14

"""
Created 2019-07-01, Bojk Berghuis



"""




def unweighted_PCA(data,n_pcs):
    print('-------------------------------------------')
    print('performing UNweighted PCA')

    metric = 'correlation'
    matrix = data.values
    matrix = np.log10(matrix + 0.1)

    # 1. whiten
    Xnorm = ((matrix.T - matrix.mean(axis=1)) / matrix.std(axis=1, ddof=0)).T

    # take care of non-varying components
    Xnorm[np.isnan(Xnorm)] = 0

    # 2. PCA
    pca = PCA(n_components=n_pcs,random_state=24870)
    # rvects columns are the right singular vectors
    rvects = pca.fit_transform(Xnorm.T)
    princdf = pd.DataFrame(index=data.columns,data=rvects)

    # 4. calculate distance matrix
    distance_matrix = squareform(pdist(rvects, metric=metric))

    return princdf,distance_matrix

def perform_tSNE(pca_df, perplexity=None):
    print('-------------------------------------------')
    print('perfoming tSNE')
    if perplexity==None:
        perplexity = 20
        print('assigned default perplexity of 20')

    x_emb = TSNE(n_components=2,perplexity=perplexity,random_state=2).fit_transform(pca_df.values)
    tsnedf = pd.DataFrame(x_emb,index=pca_df.index)
    print('tSNE done.')
    print('-------------------------------------------')
    return tsnedf

def atlas_averages_to_tsnedf(new_metadata,new_counttable,**kwargs):
    savedir = kwargs['savedir']
    date = kwargs['timestamp']
    n_pcs = kwargs['n_pcs']
    atlas = kwargs['atlas']
    cell_type_names = kwargs['CT_lut']

    #instantiate class
    sa = northstar.Averages(
            atlas=atlas,
            new_data=new_counttable,
            n_cells_per_type=kwargs['weights_atlas_cells'],
            n_features_per_cell_type=kwargs['n_features_per_cell_type'],
            n_features_overdispersed=kwargs['n_features_overdispersed'],
            n_pcs=n_pcs,
            n_neighbors=kwargs['n_neighbors'],
            n_neighbors_out_of_atlas=kwargs['n_neighbors_out_of_atlas'],
            distance_metric='correlation',
            threshold_neighborhood=kwargs['threshold_neighborhood'],
            clustering_metric='cpm',
            resolution_parameter=kwargs['resolution_parameter'],
            normalize_counts=True,
            )
    sa()

    # add new membership to metadata
    idx = new_counttable.columns
    n_fixed = len(sa.cell_types)
    new_metadata.loc[idx,'new_class'] = sa.membership
    new_metadata['new_class_renamed'] = [cell_type_names[f] if f in cell_type_names.keys() else 'NewClass_'+"{0:0=2d}".format(int(f)-n_fixed+1) if (f.isdigit()==True) else f for f in new_metadata['new_class']]

    # unweighted PCA
    cols = list(sa.cell_types)+list(new_counttable.columns)
    feature_selected_matrix = pd.DataFrame(index=sa.features_selected,columns=cols,data=sa.matrix)
    normal_PCA,distance_matrix = unweighted_PCA(feature_selected_matrix,n_pcs)

    # perform tSNE
    tsnedf = perform_tSNE(normal_PCA,20)
    tsnedf.rename(index=str,columns={0:'Dim1',1:'Dim2'},inplace=True)
    tsnedf.loc[idx,'new_membership'] = new_metadata.loc[idx,'new_class_renamed']
    tsnedf.loc[tsnedf[:n_fixed].index,'new_membership'] = tsnedf.index[:n_fixed].map(cell_type_names)

    # write params to json in new folder with date timestamp
    output_file = savedir+date+'/annotation_parameters_'+atlas+'_CellAtlasAverages_'+date+'.json'
    if not os.path.exists(os.path.dirname(output_file)):
        try:
            os.makedirs(os.path.dirname(output_file))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(output_file, 'w') as file:
        file.write(json.dumps(kwargs))
        file.close()
    # save feature matrix for later reference, e.g. making dotplots
    feature_selected_matrix.to_csv(savedir+date+'/feature_selected_matrix_'+date+'.csv')
        
        
    atlastypes = list(np.sort(tsnedf.loc[tsnedf[:n_fixed].index,'new_membership']))
    newtypes = list(set(new_metadata['new_class_renamed']).difference(atlastypes))
    celltypes = atlastypes+list(np.sort(newtypes))
    class_lut = dict(zip(celltypes,list(range(1,len(celltypes)+1))))
    tsnedf['class'] = tsnedf['new_membership'].map(class_lut)
    
    return tsnedf,celltypes,distance_matrix

def atlas_averages_annotationOnly(new_metadata,new_counttable,**kwargs):

    n_pcs = kwargs['n_pcs']
    atlas = kwargs['atlas']
    cell_type_names = kwargs['CT_lut']

    #instantiate class
    sa = northstar.Averages(
            atlas=atlas,
            new_data=new_counttable,
            n_cells_per_type=kwargs['weights_atlas_cells'],
            n_features_per_cell_type=kwargs['n_features_per_cell_type'],
            n_features_overdispersed=kwargs['n_features_overdispersed'],
            n_pcs=n_pcs,
            n_neighbors=kwargs['n_neighbors'],
            n_neighbors_out_of_atlas=kwargs['n_neighbors_out_of_atlas'],
            distance_metric='correlation',
            threshold_neighborhood=kwargs['threshold_neighborhood'],
            clustering_metric='cpm',
            resolution_parameter=kwargs['resolution_parameter'],
            normalize_counts=True,
            )
    sa()

    n_fixed = len(sa.cell_types)
    idx = list(sa.cell_types)+list(new_counttable.columns)
    annotdf = pd.DataFrame(index=idx,columns=['new_membership','class'])
    idx = new_counttable.columns
    new_metadata.loc[idx,'new_class'] = sa.membership
    new_metadata['new_class_renamed'] = [cell_type_names[f] if f in cell_type_names.keys() else 'NewClass_'+"{0:0=2d}".format(int(f)-n_fixed+1) if (f.isdigit()==True) else f for f in new_metadata['new_class']]
    annotdf.loc[idx,'new_membership'] = new_metadata.loc[idx,'new_class_renamed']
    annotdf.loc[annotdf[:n_fixed].index,'new_membership'] = annotdf.index[:n_fixed].map(cell_type_names)
    
    atlastypes = list(np.sort(annotdf.loc[annotdf[:n_fixed].index,'new_membership']))
    newtypes = list(set(new_metadata['new_class_renamed']).difference(atlastypes))
    celltypes = atlastypes+list(np.sort(newtypes))
    class_lut = dict(zip(celltypes,list(range(1,len(celltypes)+1))))
    annotdf['class'] = annotdf['new_membership'].map(class_lut)
    
    return annotdf


def atlas_subsamples_to_tsnedf(new_metadata,new_counttable,**kwargs):
    savedir = kwargs['savedir']
    date = kwargs['timestamp']
    n_pcs = kwargs['n_pcs']
    atlas = kwargs['atlas']
    cell_type_names = kwargs['CT_lut']

    #instantiate class
    no = northstar.Subsample(
            atlas=atlas,
            new_data=new_counttable,
            features=None,
            n_features_per_cell_type=kwargs['n_features_per_cell_type'],
            n_features_overdispersed=kwargs['n_features_overdispersed'],
            n_pcs=n_pcs,
            n_neighbors=kwargs['n_neighbors'],
            distance_metric='correlation',
            threshold_neighborhood=kwargs['threshold_neighborhood'],
            clustering_metric='cpm',
            resolution_parameter=kwargs['resolution_parameter'],
            normalize_counts=True,
            )
    
    no()

    # add new membership to metadata
    idx = new_counttable.columns
    n_fixed = len(no.cell_types)
    c_fixed = len(np.unique(no.cell_types))
    new_metadata.loc[idx,'new_class'] = no.membership
    new_metadata['new_class_renamed'] = [cell_type_names[f] if f in cell_type_names.keys() else 'NewClass_'+"{0:0=2d}".format(int(f)-c_fixed+1) if (f.isdigit()==True) else f for f in new_metadata['new_class']]

    # unweighted PCA
    cols = list(no.cell_names)+list(new_counttable.columns)
    feature_selected_matrix = pd.DataFrame(index=no.features_selected,columns=cols,data=no.matrix)
    normal_PCA,udistmat = unweighted_PCA(feature_selected_matrix,n_pcs)

    # perform tSNE
    tsnedf = perform_tSNE(normal_PCA,20)
    tsnedf.rename(index=str,columns={0:'Dim1',1:'Dim2'},inplace=True)
    tsnedf.loc[idx,'new_membership'] = new_metadata.loc[idx,'new_class_renamed']
    tsnedf.loc[tsnedf[:n_fixed].index,'new_membership'] = list(map(cell_type_names.get,no.cell_types))

    # write params to json in new folder with date timestamp
    output_file = savedir+date+'/annotation_parameters_'+atlas+'_CellAtlasSubsampling_'+date+'.json'
    if not os.path.exists(os.path.dirname(output_file)):
        try:
            os.makedirs(os.path.dirname(output_file))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(output_file, 'w') as file:
        file.write(json.dumps(kwargs))
        file.close()
        
        
    atlastypes = list(np.sort(tsnedf.loc[tsnedf[:n_fixed].index,'new_membership'].unique()))
    newtypes = list(set(new_metadata['new_class_renamed']).difference(atlastypes))
    celltypes = atlastypes+list(np.sort(newtypes))
    
    return tsnedf,celltypes





def make_pairs(distmat,threshold,max_neighbors):
    """
    returns the edges (based on distance matrix indices) with the closest distance
    that exist below a certain cutoff threshold, for a maximum of 'max_neighbors' edges
    """
    corr = str((1-threshold)*100)
    print('---------------------------------------')
    print('Making list of edges with '+corr+'% correlation and up')
    print('Max '+str(max_neighbors)+' edges per cell.')
    pairs = []
    for cell in distmat.index:
        temp = distmat.loc[cell].sort_values()[1:max_neighbors]
        neigh = temp[temp<threshold].index
        if len(neigh)>0:
            for pa in neigh:
                pairs.append((cell,pa))
    print('Found '+str(len(pairs))+' edges.')
    print('---------------------------------------')
    return pairs

def make_pairdf(dist_matrix,NN,tsne_df,colname):
    """ returns a dataframe with pairs and some properties
    such as inter or intra class edge, eucledian distance between the two in the tsne plane"""
    xcol = colname+'1'
    ycol = colname+'2'
    pairs = make_pairs(dist_matrix,2,NN)
    pair_df = pd.DataFrame(pairs)
    for pair in pair_df.index:
        cell1 = pair_df.loc[pair,0]
        cell2 = pair_df.loc[pair,1]
        if type(dist_matrix.index[0])==int:
            class1 = tsne_df.iloc[cell1]['new_membership']
            class2 = tsne_df.iloc[cell2]['new_membership']
        else:
            class1 = tsne_df.loc[cell1,'new_membership']
            class2 = tsne_df.loc[cell2,'new_membership']
        if class1==class2:
            pair_df.loc[pair,'edge_type'] = 'intra_class'
        else:
            pair_df.loc[pair,'edge_type'] = 'inter_class'
        pair_df.loc[pair,'correlation'] = 1 - dist_matrix.loc[cell1,cell2]
        pair_df.loc[pair,'distance'] = dist_matrix.loc[cell1,cell2]
        if  type(dist_matrix.index[0])==int:
            xy1 = tsne_df.iloc[cell1][[xcol,ycol]]
            xy2 = tsne_df.iloc[cell2][[xcol,ycol]]
        else:
            xy1 = tsne_df.loc[cell1][[xcol,ycol]]
            xy2 = tsne_df.loc[cell2][[xcol,ycol]]
        pair_df.loc[pair,'edge_length'] =  np.sqrt((xy2[0]-xy1[0])**2+(xy2[1]-xy1[1])**2)
    return pair_df


