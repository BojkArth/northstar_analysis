#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist
from fastcluster import linkage
#from polo import optimal_leaf_ordering

"""
Bojk Berghuis
created 19-07-2019

Main functionality: Makes a dot plot from a (feature selected) matrix of genes X cells (and/or cell types)
                                In support of semiAnnotate algorithm

INPUT:                     matrix,
                                tsnedf (metadata from semiAnnotate containing classes),
                                number_of_genes,
                                kwargs (see below)

OUT:                        df of all overdispersed genes
                                list of ordered rows
                                list of ordered columns
                                df of len(number_of_genes) x classes containing top overdispersed genes per class

            keys = ['left_column', 'right_column','figure_name','savedir','close_plot']
            values = ['original_membership','new_membership','spyros_classification_SelfEgdes_top5'
                                ,'../leidenalg/Datasets/Darmanis_Brain/'+date+'/',True]
            kwargs = dict(zip(keys,values))
            genePanel,rows,cols,gene_table = make_dotplot(matrix,tsnedf,5,**kwargs)

If you just want to order a correlation matrix ('Pearson correlation')

        use:    order_correlation_matrix()


"""

def make_dotplot(feat_sel_matrix,tsne_df,numgenes,colorlist=None,**kwargs):
    column_name = kwargs['right_column']
    gene_panel,panel_overdisp,genes_OD,avg_exp = make_top_genes(feat_sel_matrix,tsne_df,column_name,numgenes)
    savedir = kwargs['savedir']
    savename = kwargs['figure_name']
    genes_OD.to_csv(savedir+'gene_table_'+savename+'.csv')

    #use this if you want optimal ordered classes:
    #index,cols = ordered_classes(panel_overdisp)

    # use this if you just want the diagonal:
    cols = genes_OD.columns.sort_values()
    index = genes_OD[cols[0]]
    for celltype in cols[1:]:
        index = index.append(genes_OD[celltype]).reset_index(drop=True)
    index = index[::-1]

    kwargs['figure_name'] = kwargs['figure_name']+'_overdispersed_'
    if colorlist is not None:
        plot_dot(panel_overdisp,index,cols,colorlist,**kwargs)
    else:
        plot_dot(panel_overdisp,index,cols,**kwargs)
    return panel_overdisp,index,cols,genes_OD,avg_exp


def _optimal_order(data, **kwargs):
    """ Optimal leaf ordering
        **kwargs passed to pdist e.g. metric='correlation'
    """
    d = pdist(data, **kwargs)
    optimal_order = linkage(d, method='average', optimal_ordering=True)
    #NOTE: recent scipy includes this
    #optimal_order = optimal_leaf_ordering(link, d)
    return optimal_order

def make_top_genes(feat_sel_matrix,tsne_df,column_name,numgenes):
    cols = tsne_df[column_name].unique()
    genes = pd.DataFrame(index=range(numgenes),columns=cols)
    class_exp = pd.DataFrame(index=feat_sel_matrix.index,columns=cols)
    genes_OD = genes.copy()
    for group in cols:
        members = tsne_df[tsne_df[column_name]==group].index
        class_exp[group] = feat_sel_matrix.loc[:, members].T.mean()
        # this takes the top expressing genes per class in the feature selected matrix
        genes[group] = feat_sel_matrix[members].mean(axis=1).sort_values(ascending=False)[:numgenes].index

    # this retrieves the overdispersed genes in a class, when compared to the average
    features = set()
    for group in cols:
        ge1 = class_exp[group]
        ge2 = (class_exp.sum(axis=1) - ge1).divide(len(class_exp.T) - 1)
        fold_change = np.log2(ge1 + 0.1) - np.log2(ge2 + 0.1)
        #use this for large datasets:
        """markers = np.argpartition(fold_change.values, -numgenes)[-numgenes:]
        genelista = class_exp.iloc[markers].index
        features_argp |= set(genelista)
        genes_OD_argp[group] = genelista"""
        # use this in any other situation (as the genes are automatically ranked from most to least overdispersed)
        markers = fold_change.sort_values(ascending=False)[:numgenes]
        genelist = markers.index
        features |= set(genelist)
        genes_OD[group] = genelist

    unique_genes = set()
    for c in genes.columns:
        unique_genes |= set(genes[c])
    panel = pd.DataFrame(index=unique_genes,columns=cols)
    panel_OD = pd.DataFrame(index=features,columns=cols)
    for group in cols:
        members = tsne_df[tsne_df[column_name]==group].index
        panel[group] = feat_sel_matrix.loc[panel.index,members].mean(axis=1)
        panel_OD[group] = feat_sel_matrix.loc[panel_OD.index,members].mean(axis=1)
    return panel,panel_OD,genes_OD,class_exp

def ordered_classes(panel):
    row_link = _optimal_order(panel.T.corr(), metric='correlation')
    col_link = _optimal_order(panel.corr(), metric='correlation')
    #cg = sns.clustermap(panel.T.corr(), row_linkage=row_link, col_linkage=row_link,figsize=(20,55),xticklabels=False)
    cg = sns.clustermap(np.log2(panel+0.001),row_linkage=row_link,col_linkage=col_link)
    plt.close()
    rows = cg.dendrogram_row.reordered_ind
    columns = cg.dendrogram_col.reordered_ind

    row_names = list(panel.iloc[rows].index)
    col_names = list(panel.T.iloc[columns].index)
    return row_names, col_names

def order_correlation_matrix(matrix):
    row_link = _optimal_order(matrix, metric='correlation')
    cg = sns.clustermap(matrix,row_linkage=row_link,col_linkage=row_link)
    plt.close()
    rows = cg.dendrogram_row.reordered_ind
    row_names = list(matrix.iloc[rows].index)
    return row_names

def plot_dot(panel,rows,columns,colorlist=None,**kwds):
    figname = kwds['figure_name']
    savedir = kwds['savedir']
    xgrid = list(range(len(panel.columns)))
    ygrid = np.ones(len(xgrid))
    height = 20
    width = 7
    if len(panel)<50:
        f = plt.figure(figsize=(width,height))
    else:
        f = plt.figure(figsize=(20,len(panel)))
    i=0
    for gene in panel.loc[rows].index:
        temp = panel[panel.index==gene].loc[:,columns]
        plt.scatter(xgrid,ygrid*i,s=temp.divide(1),alpha=.8)
        i+=1
    plt.yticks(np.arange(len(panel)),rows)
    plt.xticks(np.arange(len(panel.T)),columns,rotation=90)
    f.savefig(savedir+'dotplot_'+figname+'.png')
    f.savefig(savedir+'dotplot_'+figname+'.pdf')
    plt.close(f)
    if len(panel)<50:
        f = plt.figure(figsize=(width,height))
    else:
        f = plt.figure(figsize=(20,len(panel)))
    i=0
    for gene in panel.loc[rows].index:
        temp = panel[panel.index==gene].loc[:,columns]
        if colorlist is not None:
            plt.scatter(xgrid,np.ones(len(xgrid))*i,s=np.log2(temp).multiply(70),c=colorlist,alpha=.8)
        else:
            plt.scatter(xgrid,np.ones(len(xgrid))*i,s=np.log2(temp).multiply(70),alpha=.8)
        i+=1
    plt.gcf().subplots_adjust(left=.28,right=0.98,bottom=.2)
    plt.yticks(np.arange(len(panel)),rows)
    plt.xticks(np.arange(len(panel.T)),columns,rotation=90)
    f.savefig(savedir+'dotplot_log2_'+figname+'.png')
    f.savefig(savedir+'dotplot_log2_'+figname+'.pdf')
    plt.close(f)

    
