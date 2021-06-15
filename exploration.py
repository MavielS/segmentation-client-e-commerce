import pandas as pd
from tabulate import tabulate
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

import squarify
import matplotlib.colors

def analyse_forme(df,thresh_na=60, all=False):
    
    statList = {'Taille':[df.size],'Nb lignes':[df.shape[0]],
                'Nb colonnes':[df.shape[1]],
                '% de NaN':[round(100.0 * (df.isna().sum().sum())/df.size,2)],
                'Nb duplicats':df.duplicated().sum()}
    statsValues = pd.DataFrame().from_dict(statList, orient='columns')
    print(tabulate(statsValues, headers = 'keys', tablefmt = 'psql'))    

    if(all):
        # print(df.head(10))
    
        plt.figure(figsize=(16,8))

        ax1 = plt.subplot(1,2,1)
        vc = df.dtypes.value_counts()

        # create a color palette, mapped to these values
        cmap = matplotlib.cm.Blues
        mini=min(vc)
        maxi=max(vc)
        norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
        colors = [cmap(norm(value)) for value in vc]

        squarify.plot(ax = ax1, sizes=list(vc.values), label=pd.Series(vc.keys()).to_string(index=False).replace(" ", "").splitlines(), alpha=.8, value=round(vc/vc.sum(),2), pad=0.05, color=colors,  text_kwargs={'fontsize': 22, 'fontfamily' : 'sans-serif'})
        plt.axis('off')
        ax1.set_title('Répartition valeurs quantitatives/qualitatives',fontsize=20,weight='bold')

        ax2 = plt.subplot(1,2,2)
        perc = (df.isnull().sum()/df.shape[0])*100
        perc = perc.sort_values(ascending=False)
        perc.index = np.arange(0,df.shape[1],1)

        ax2 = sns.barplot(x=perc.index,y=perc, palette=sns.dark_palette("#69d", reverse=True))
        plt.axhline(y=thresh_na, color='r', linestyle='-')
        plt.text(len(df.isnull().sum()/len(df))/1.7, thresh_na+12.5, 'Columns with more than %s%s missing values' %(thresh_na, '%'), fontsize=12,weight='bold', color='crimson',
             ha='left' ,va='top')
        plt.text(len(df.isnull().sum()/len(df))/1.7, thresh_na - 5, 'Columns with less than %s%s missing values' %(thresh_na, '%'), fontsize=12,weight='bold', color='blue',
             ha='left' ,va='top')

        ax2.set_title('NaN par colonnes',fontsize=20, weight='bold')
        ax2.set_xlabel('Colonnes',fontsize=20)
        ax2.set_ylabel('% de NaN',fontsize=20)
        ax2.set_xticks(np.arange(0,df.shape[1],5))
        ax2.set_yticks(np.arange(0,101,5))

        plt.show()

        

def premier_quartile(data_frame,colonne):
    """Pour une variables quantitative. Retourne la valeur du premier quartile. Colonne est le nom de la colonne."""
    return data_frame[colonne].quantile(q=0.25)


def troisieme_quartile(data_frame,colonne):
    """Pour une variables quantitative. Retourne la valeur du troisième quartile. Colonne est le nom de la colonne"""
    return data_frame[colonne].quantile(q=0.75)


def inter_quartile(data_frame,colonne):
    """Retourne l'écart inter-quartile."""
    return troisieme_quartile(data_frame,colonne)-premier_quartile(data_frame,colonne)


def outliers_IQR(data, col, alpha=1.5):
    '''
    Retourne les outliers par la méthode de l'IQR.
    @return borne_inf, borne_sup
    '''
    iqr = inter_quartile(data,col)
    q1 = premier_quartile(data,col)
    q3 = troisieme_quartile(data,col)
    
    outliers_inf = data[data[col] < q1 - alpha*iqr]
    outliers_sup = data[data[col] > q3 + alpha*iqr]
    
    print("Q1: {}; Q3: {}; Outliers inf: {}; Outliers sup: {} ".format(q1,q3, len(outliers_inf), len(outliers_sup)))
    
    if(outliers_inf.empty and outliers_sup.empty):
        return outliers_inf, outliers_sup
    
    else:
        tmp = data.copy()

        if(not outliers_inf.empty):
            tmp = tmp.drop(outliers_inf.index)
        if(not outliers_sup.empty):
            tmp = tmp.drop(outliers_sup.index)

        # Calcul des nouveaux outliers résultants de la suppression des anciens
        to_add_inf, to_add_sup = outliers_IQR(tmp, col, alpha)
        outliers_inf =  outliers_inf.append(to_add_inf)
        outliers_sup =  outliers_sup.append(to_add_sup)

        return outliers_inf, outliers_sup
    
def mod_z(col: pd.Series, alpha: float=0.6745) -> pd.Series:
    '''
    Renvoie le Z-score modifié de notre variable col
    '''
    med_col = col.median()
    med_abs_dev = (np.abs(col - med_col)).median()
    mod_z = alpha * ((col - med_col) / med_abs_dev)
    return np.abs(mod_z)

def outliers_mod_z(data: pd.DataFrame, col: str, thresh: int=3, alpha: float=0.6745):
    '''
    Retourne les outliers par le score z-score modifié.
    @return borne_inf, borne_sup
    '''
    z = mod_z(data[col], alpha)
    
    outliers_inf = data[z < -thresh]
    outliers_sup = data[z >  thresh]
    
    print("Outliers inf: {}; Outliers sup: {} ".format(len(outliers_inf), len(outliers_sup)))
    
    if(outliers_inf.empty and outliers_sup.empty):
        return outliers_inf, outliers_sup
    
    else:
        tmp = data.copy()

        if(not outliers_inf.empty):
            tmp = tmp.drop(outliers_inf.index)
        if(not outliers_sup.empty):
            tmp = tmp.drop(outliers_sup.index)

        # Calcul des nouveaux outliers résultants de la suppression des anciens
        to_add_inf, to_add_sup = outliers_mod_z(tmp, col, thresh, alpha)
        outliers_inf =  outliers_inf.append(to_add_inf)
        outliers_sup =  outliers_sup.append(to_add_sup)

        return outliers_inf, outliers_sup
    
def mesure_forme(data: pd.DataFrame, col: str, bins=None):  
    '''
    Nous affiche un bel histogramme (et son boxplot associé) de la variable col
    '''
    if bins == None:
        # Sturge's Rule
        bins = int(1 + 3.322*np.log(len(data[col].unique())))
        print(f'Nb de bins optimal estimé: {bins}')
    
    dataMean = round(data[col].dropna().mean(), 2)
    dataMedian = round(data[col].dropna().median(), 2)
    dataStd = round(data[col].dropna().std(),  2)
    dataSkew = round(data[col].dropna().skew(), 2)
    dataKurt = round(data[col].dropna().kurt(), 2)
    
    fig, (ax_hist, ax_box) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (1, 0.2)}, figsize=(12,8))
    
    sns.histplot(ax=ax_hist, data=data[col].dropna(),kde=True, bins=bins, color=sns.color_palette('deep')[0])
    
    ax_hist.plot([], [], ' ', label=f'Skewness = {dataSkew}')
    ax_hist.plot([], [], ' ', label=f'Kurtosis = {dataKurt}')
    ax_hist.plot([], [], ' ', label=f'Moyenne  = {dataMean}')
    ax_hist.plot([], [], ' ', label=f'Variance = {dataStd}')
    ax_hist.plot([], [], ' ', label=f'Mediane = {dataMedian}')
    
    ax_hist.legend( loc='upper right', borderaxespad=0., fontsize='large')    
    ax_hist.set_title(f'Distribution {col}',fontsize=20)

    meanprops = {'marker':'o', 'markeredgecolor':'black','markerfacecolor':'firebrick'}
    
    sns.boxplot(ax=ax_box, x=data[col].dropna(), showmeans=True, meanprops=meanprops, color=sns.color_palette('deep')[0])
    
    ax_box.set_xlabel("")
    ax_box.set_ylabel("")
        
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)    
    
    plt.show()
    
def data_trimming(data, quantile_sup=0.995, quantile_inf=0.005):
    '''
    Retourne des valeurs extrêmes du dataset de l'ensemble des colonnes quantitatives de data
    Ces valeurs extremes sont celles > à quantile_sup et < à quantile_inf
    '''
    tmp = data.copy()

    to_del = []
    for col in tmp.select_dtypes(exclude = ['object','bool']).columns.tolist() :
        
        to_del.extend(tmp.loc[tmp[col] > tmp[col].quantile(quantile_sup)].index)
        to_del.extend(tmp.loc[tmp[col] < tmp[col].quantile(quantile_inf)].index)
    
    return tmp.loc[to_del]
    

    
    
    
    
    