import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import colors, ticker
from matplotlib.colors import LogNorm
from bokeh import palettes
import holoviews as hv
import hvplot.pandas 
import scipy
import torch.nn.functional as F
from matplotlib.colors import LogNorm, Normalize
import seaborn as sns
sns.set_theme(style="ticks")
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, accuracy_score, r2_score
import torch
import torchmetrics
from sklearn.metrics import ConfusionMatrixDisplay

from data.data_utils import get_height_level_range
from data.data_module import LogTransform, LogTransform2D

hv.extension('matplotlib')

plt_kwargs_commons = {'var_range': 
                      {'iwc': (1e-07, 0.001), 
                       'nice': (1e2, 1e6)},
                      'var_range_log': 
                      {'iwc': (-7, -3), 
                       'nice': (2, 6)},
                      'axis_title':
                       {'iwc': 'IWC [kg m⁻³]',
                        'nice': 'Nice [m⁻³]'}}

y_hat_color = "#30A2DA"
dardar_color = "#FD654B"

def cloud_occurance_per_height_level(y_hat, 
                                     dardar, 
                                     y_hat_cube=None,
                                     cloud_thres=0,
                                     upper_thres=1e9,
                                     height_levels:np.ndarray=get_height_level_range(1680,16980,step=60),
                                     plt_kwargs=dict(invert=True,
                                                     title="cloud ice occurance per height level",
                                                     xlim=[4000,17000],
                                                     ylim=[0,15],
                                                     ylabel="Altitude [m]", 
                                                     xlabel="Cloud ice occurance [%]",
                                                     legend="right",
                                                     aspect=0.7,     
                                                     grid=True,   
                                                     fontsize={"title":24, "xlabel":20,"ylabel":20,"legend":20,"ticks":15},
                                                     linewidth=3,
                                                     linestyle=["-","-",":"],
                                                     shared_axes=False)):

    # y_hat cloud occurance                                 
    y_hat_cloud_occurance = (y_hat > cloud_thres) & (y_hat < upper_thres)
    y_hat_cloud_occurance = y_hat_cloud_occurance.sum(dim=0) / y_hat_cloud_occurance.shape[0]
                                     
    # dardar cloud occurance
    dardar_cloud_occurance = (dardar > cloud_thres) & (dardar < upper_thres)
    dardar_cloud_occurance = dardar_cloud_occurance.sum(dim=0) / dardar_cloud_occurance.shape[0] 
                 
    # create df with percentages                                     
    cloud_occurance_df = pd.DataFrame([dardar_cloud_occurance.cpu().numpy(),y_hat_cloud_occurance.cpu().numpy()],
                                  index=["DARDAR","IceCloudNet"],
                                  columns=height_levels).T * 100 
    
                                     
    # add cube data if given (costs a lot of memory)
    if y_hat_cube is not None:
        cube_pred_cloud_occurance = (y_hat_cube > cloud_thres) & (y_hat_cube < upper_thres)
        cube_pred_cloud_occurance = cube_pred_cloud_occurance.sum(dim=(0,2,3)) / (cube_pred_cloud_occurance.shape[0] * cube_pred_cloud_occurance.shape[2] * cube_pred_cloud_occurance.shape[3])
        cloud_occurance_df["IceCloudNet \nwhole domain"] = cube_pred_cloud_occurance * 100

    p = cloud_occurance_df.hvplot.line(y=list(cloud_occurance_df.columns),**plt_kwargs,color=[dardar_color,y_hat_color,"grey"])
    # plt.tight_layout()
    return cloud_occurance_df, p                                 

def iwc_per_height_df(y: torch.Tensor,
                      height_levels:np.ndarray=get_height_level_range(1680,16980,step=60),
                      color="#30A2DA",
                      min_n_per_height=0.0005,
                      plt_q10=False,
                      incloud=True,
                      target_variable="iwc",
                      plt_kwargs=dict(logy=False,
                                      logx=True,
                                      ylim=(1e-7,1e-3),
                                      xlim=(4000,17000),
                                      ylabel="Altitude [m]",
                                      xlabel="IWC [kg m⁻³]",
                                      xticks=np.logspace(-7,-3,num=5),
                                      fontscale=1,aspect=1.6,
                                      fontsize={"title":24, "xlabel":20,"ylabel":20,"legend":20,"ticks":15},
                                      legend="top_right")):
    
    
    
    plt_kwargs["ylim"] = plt_kwargs_commons["var_range"][target_variable]
    plt_kwargs["xticks"] = np.logspace(*plt_kwargs_commons["var_range_log"][target_variable],num=5)
    plt_kwargs["xlabel"] = plt_kwargs_commons["axis_title"][target_variable]
    
    
    # print(plt_kwargs)
    
    # mask non-cloud pixels
    if incloud:
        y_in_cloud = y.masked_fill(y==0, torch.nan)
    else:
        y_in_cloud = y
    
    # calculate statistics
    y_count, y_mean, y_median, y_q10,y_q25, y_q75,y_q90 = y_in_cloud.cpu().isfinite().sum(dim=0), y_in_cloud.cpu().nanmean(dim=0), y_in_cloud.cpu().nanmedian(dim=0).values, y_in_cloud.cpu().nanquantile(0.1,dim=0),y_in_cloud.cpu().nanquantile(0.25,dim=0), y_in_cloud.cpu().nanquantile(0.75,dim=0), y_in_cloud.cpu().nanquantile(0.9,dim=0)
    
    height_iwc_df = pd.DataFrame([y_count.cpu().numpy(), y_mean.cpu().numpy(), y_median.cpu().numpy(), y_q10.cpu().numpy(),y_q25.cpu().numpy(), y_q75.cpu().numpy(),y_q90.cpu().numpy()],
                                   index=["ncloud_pixels",f"{target_variable}_mean", f"{target_variable}_median",f"{target_variable}_q10", f"{target_variable}_q25", f"{target_variable}_q75",f"{target_variable}_q90"],
                                   columns=height_levels).T
    
    min_n_per_height = height_iwc_df.ncloud_pixels.sum()*min_n_per_height # calculate as percentage of total cloud pixels
    print("min_n_per_height", min_n_per_height)
    plt_df = height_iwc_df.query(f"ncloud_pixels>{min_n_per_height}")
    
    
#     med_plt= plt_df.hvplot.line(y="iwc_median",invert=True).opts(**plt_kwargs,color=color)
#     q_plts = plt_df.hvplot.area(y="iwc_q25",y2="iwc_q75",invert=True).opts(**plt_kwargs,alpha=0.2,color=color)
#     q_plts2 =plt_df.hvplot.area(y="iwc_q10",y2="iwc_q90",invert=True).opts(**plt_kwargs,alpha=0.1,color=color)
    
    med_plt= plt_df.hvplot.line(y=f"{target_variable}_median",invert=True,**plt_kwargs,color=color)
    q_plts = plt_df.hvplot.area(y=f"{target_variable}_q25",y2=f"{target_variable}_q75",invert=True,**plt_kwargs,alpha=0.2,color=color)
    q_plts2 =plt_df.hvplot.area(y=f"{target_variable}_q10",y2=f"{target_variable}_q90",invert=True,**plt_kwargs,alpha=0.1,color=color)
        
    p = med_plt * q_plts                  
    if plt_q10:
        p = p * q_plts2
    
    return height_iwc_df, p

def iwc_in_cloud_distribution(y_hat, 
                              dardar,
                              cloud_thres=0,
                              target_variable="iwc"):
    
    if target_variable == "iwc":
        target_transform = LogTransform(scaler=1e7)
    elif target_variable == "nice":
        target_transform = LogTransform(scaler=1e-2)
    
    xlim = plt_kwargs_commons["var_range"][target_variable]
    xlim = (xlim[0]*1e-1,xlim[1]*1e1) # extend by 1 oom
    
    
    y_hat_1d = torch.flatten(y_hat)
    dardar_1d = torch.flatten(dardar)

    y_hat_1d = y_hat_1d.cpu().numpy()
    dardar_1d = dardar_1d.cpu().numpy()
    
    fig, ax = plt.subplots(1,2,figsize=(14,5))
    
    sns.histplot(data=target_transform(y_hat_1d[y_hat_1d>cloud_thres]),log_scale=(False,False),bins=50,stat="frequency",ax=ax[0],alpha=0.25,color=y_hat_color,label="test")
    sns.histplot(data=target_transform(dardar_1d[dardar_1d>cloud_thres]),log_scale=(False,False),bins=50,stat="frequency",ax=ax[0],alpha=0.25,color=dardar_color)
    ax[0].set_xlim(0,5)
    ax[0].title.set_text('predicted vs. target IWC [log transformed]')
    ax[0].set_xlabel("IWC [kg m⁻³] log transformed")

    sns.histplot(data=y_hat_1d[y_hat_1d>cloud_thres],log_scale=(True,False),bins=50,stat="frequency",ax=ax[1],alpha=0.25,color=y_hat_color)
    sns.histplot(data=dardar_1d[dardar_1d>cloud_thres],log_scale=(True,False),bins=50,stat="frequency",ax=ax[1],alpha=0.25,color=dardar_color)
    ax[1].set_xlim(*xlim)
    ax[1].title.set_text('predicted vs. target IWC [kg m⁻³]')
    ax[1].set_xlabel("IWC [kg m⁻³]")

    plt.tight_layout()
    
    return fig

def metrics_per_level(y_hat, 
                      dardar,
                      y_hat_nice=None,
                      dardar_nice=None,
                      target_transform=LogTransform(scaler=1e7),
                      nice_target_transform=LogTransform(scaler=1e-2),
                      cloud_thres=0,
                      height_levels=get_height_level_range(),
                      n_level_aggregation=16,
                      show_mae=False,
                      plt_kwargs=dict(invert=True,
                                     xlim=(4000,17000), 
                                     aspect=1,   
                                     color=list(palettes.Dark2_5),
                                     linewidth=3,
                                     ylim=(0,1),
                                     fontsize={"title":24, "xlabel":20,"ylabel":20,"legend":20,"ticks":15},
                                     title="Performance metrics per height level",
                                     ylabel="Altitude [m]",                  
                                     legend="right",
                                     grid=True),
                     mae_plt_kwargs=dict(invert=True,
                                     xlim=(4000,17000),    
                                     color=list(palettes.Accent3),
                                     linewidth=3,
                                     logx=True, 
                                     ylim=(1e-8,1e-3),
                                     xticks=np.logspace(-8,-3,5), 
                                     # ylim=(0,1),
                                     fontsize={"title":24, "xlabel":20,"ylabel":20,"legend":20,"ticks":15},
                                     title="MAE per height level",
                                     xlabel="Altitude [m]",
                                     ylabel="MAE", 
                                     legend="right",
                                     grid=True)):
    """create line plot with height on yaxis, and metric on x-axis
    
    instead of calculating it per each heightlevel, combine `n` height levels
    """
    ps = []
    rs = []
    acc = []
    r2 = []
    r2_nice = []
    mae = []
    df_index = []
    
    n_height_bins = int(np.ceil(y_hat.shape[1] / n_level_aggregation))
    
    for height_bin in range(n_height_bins-1):
        y_hat_1d = torch.flatten(y_hat[:,height_bin*n_level_aggregation:(height_bin+1)*n_level_aggregation])
        dardar_1d = torch.flatten(dardar[:,height_bin*n_level_aggregation:(height_bin+1)*n_level_aggregation])
        
        #if y_hat_nice is not None:
        #    y_hat_1d_nice = torch.flatten(y_hat_nice[:,height_bin*n_level_aggregation:(height_bin+1)*n_level_aggregation])
        #    dardar_1d_nice = torch.flatten(dardar_nice[:,height_bin*n_level_aggregation:(height_bin+1)*n_level_aggregation]) 
        #    r2_nice.append(r2_score(nice_target_transform(dardar_1d_nice), nice_target_transform(y_hat_1d_nice)))
        
        y_hat_1d = y_hat_1d.cpu().numpy()
        dardar_1d = dardar_1d.cpu().numpy()

        mae.append(mean_absolute_error(dardar_1d[dardar_1d>0],y_hat_1d[dardar_1d>0]))
        ps.append(precision_score(dardar_1d>cloud_thres,y_hat_1d>cloud_thres))
        rs.append(recall_score(dardar_1d>cloud_thres,y_hat_1d>cloud_thres))
        acc.append(accuracy_score(dardar_1d>cloud_thres,y_hat_1d>cloud_thres))
        r2.append(r2_score(target_transform(dardar_1d), target_transform(y_hat_1d)))
        df_index.append(height_levels[height_bin*n_level_aggregation])
        
    df = pd.DataFrame(np.array([ps,rs,acc,r2]).T,columns=["precision", "recall", "accuracy","R²"], index=df_index)
    if y_hat_nice is not None:
        df["nice_R²"] = r2_nice
    df = df.round(2)
    p = df.hvplot.line(**plt_kwargs)
    if show_mae:
        df["MAE"] = mae
        mae_p = df["MAE"].hvplot.line(**mae_plt_kwargs)
        p = (mae_p+p).opts(tight=False,fig_inches=5.75)
    
    return df, p


def iwc_vs_iwc_plt(y_hat, dardar, target_variable="iwc",target_transform=LogTransform(scaler=1e7),plt_kwargs=dict(clim=(1e2,1e5),aspect=1,fontsize={"title":24, "xlabel":20,"ylabel":20,"clabel":20,"legend":20,"ticks":15},ylabel="IceCloudNet IWC [kg m⁻³]",xlabel="DARDAR IWC [kg m⁻³]",clabel="# pixel",cmap=palettes.Blues9[::-1],gridsize=100,min_count=50)):
    """in cloud reference vs dardar"""
    plt_kwargs["xlim"] = plt_kwargs_commons["var_range"][target_variable]
    plt_kwargs["ylim"] = plt_kwargs_commons["var_range"][target_variable]
    plt_kwargs["xlabel"] = f'DARDAR {plt_kwargs_commons["axis_title"][target_variable]}'
    plt_kwargs["ylabel"] = f'IceCloudNet {plt_kwargs_commons["axis_title"][target_variable]}'
    
    y_hat_1d = torch.flatten(y_hat)
    dardar_1d = torch.flatten(dardar)

    # stack y_hat, dardar to same tensor
    stacked_1d = torch.vstack((dardar_1d,y_hat_1d)).T
    stacked_1d_incloud_dardar = stacked_1d[stacked_1d[:,0]>0]
    stacked_1d_incloud =  stacked_1d[stacked_1d.min(dim=1).values>0]#[:10000000] # select pixels with both cloudy

    df = pd.DataFrame(stacked_1d_incloud.cpu().numpy(),columns=["Dardar","IceCloudNet"])

    iwc_plt = df.query("IceCloudNet >0").hvplot.hexbin(y="Dardar",x="IceCloudNet",invert=True,logx=True,logy=True,logz=True).opts(**plt_kwargs)

    iwc_plt = iwc_plt * hv.Curve([[0,0],[1e7,1e7]]).opts(color="grey",linewidth=3,linestyle="--")

    mae_all_cloud = torch.abs(stacked_1d[:,0] - stacked_1d[:,1]).mean()
    mae_incloud = torch.abs(stacked_1d_incloud_dardar[:,0] - stacked_1d_incloud_dardar[:,1]).mean()
    mae_incloud_log = torch.abs(target_transform(stacked_1d_incloud[:,0]) - target_transform(stacked_1d_incloud[:,1])).mean()
    mape = torchmetrics.functional.mean_absolute_percentage_error(stacked_1d_incloud[:,1],stacked_1d_incloud[:,0])
    mape_log = torchmetrics.functional.mean_absolute_percentage_error(target_transform(stacked_1d_incloud[:,1]),target_transform(stacked_1d_incloud[:,0]))
    
    # calculate conditional mean
    conditional_mean_cond_icn = target_transform.inverse_transform(target_transform(df).round(1).groupby("IceCloudNet").describe()["Dardar"][["std","mean"]].reset_index()).query("IceCloudNet >0") # type: ignore
    conditional_mean_cond_icn["lower"] = conditional_mean_cond_icn["mean"] - conditional_mean_cond_icn["std"]
    conditional_mean_cond_icn["upper"] = conditional_mean_cond_icn["mean"] + conditional_mean_cond_icn["std"]
    cond_mean_plts = conditional_mean_cond_icn.hvplot.line(y="mean",x="IceCloudNet",invert=True,legend=True,logx=True,logy=True,color=palettes.Reds3[1],linewidth=3) * conditional_mean_cond_icn.hvplot.area(x="IceCloudNet",y="lower",y2="upper",invert=True,alpha=0.5).opts(facecolor=palettes.Reds3[1])
    
    
    
    # p = iwc_plt * hv.Text(plt_kwargs["xlim"][1]*0.1e-2,plt_kwargs["xlim"][0]*5e0,f'MAE: {mae_incloud:.1e}',fontsize=20,halign="left") * hv.Text(plt_kwargs["xlim"][1]*0.1e-2,plt_kwargs["xlim"][0]*2.5e0,f'MAE log10: {mae_incloud_log:.2f}',fontsize=20,halign="left")
    p = iwc_plt * cond_mean_plts * hv.Text(plt_kwargs["xlim"][0]*5e3,plt_kwargs["xlim"][0]*2.5e0,f'MAE log10: {mae_incloud_log:.2f}',fontsize=20,halign="left")
    # p = iwc_plt * hv.Text(plt_kwargs["xlim"][1]*0.1e-2,plt_kwargs["xlim"][0]*5e0,f'MAPE: {mape:.2%}',fontsize=20,halign="left") * hv.Text(plt_kwargs["xlim"][1]*0.1e-2,plt_kwargs["xlim"][0]*2.5e0,f'MAPE log10: {mape_log:.2%}',fontsize=20,halign="left")
    
    return df, p

def get_metrics(y_hat, dardar,cloud_thres=0,target_transform=LogTransform(scaler=1e7)):
    y_hat_1d = torch.flatten(y_hat)
    dardar_1d = torch.flatten(dardar)
    mae = float(F.l1_loss(y_hat_1d,dardar_1d,reduction="mean"))
    mae_incloud = float(F.l1_loss(y_hat_1d[dardar_1d>0],dardar_1d[dardar_1d>0],reduction="mean"))
    # classification metrics
    ps = precision_score(dardar_1d>cloud_thres,y_hat_1d>cloud_thres)
    rs = recall_score(dardar_1d>cloud_thres,y_hat_1d>cloud_thres)
    acc = accuracy_score(dardar_1d>cloud_thres,y_hat_1d>cloud_thres)
    # regression metrics
    r2_log = r2_score(target_transform(dardar_1d).numpy(),target_transform(y_hat_1d).numpy())
    r2 = r2_score(dardar_1d.numpy(),y_hat_1d.numpy())
    corr = np.corrcoef(dardar_1d, y_hat_1d)[0][1]
    # in cloud regression metrics (based on predicted in cloud values)
    r2_incloud = r2_score(dardar_1d[y_hat_1d>0].numpy(),y_hat_1d[y_hat_1d>0].numpy())
    corr_incloud = np.corrcoef(dardar_1d[y_hat_1d>0], y_hat_1d[y_hat_1d>0])[0][1]
    
    metric_dict = dict(precision=ps,recall=rs,accuracy=acc,r2_log=r2_log,r2=r2,corr=corr,r2_incloud=r2_incloud,corr_incloud=corr_incloud,mae=mae,mae_incloud=mae_incloud)
    
    return metric_dict


def horizontal_cloud_cover(dardar_cloud_cover_data, 
                           y_hat_cloud_cover_data, 
                           hex_plt_kwargs=dict(kind="hex",height=7,joint_kws={'gridsize':20, 'bins':'log',"mincnt":50,"cmap":plt.get_cmap("Blues",10)},marginal_kws={'bins':50}),
                           line_plt_kwargs=dict(invert=True, title="IceCloudNet height dependent \n cloud cover skill ",height=450, aspect=0.9, color=list(palettes.Accent3),linewidth=3,ylim=(0,1),xlim=(4000,17000),fontscale=1,xlabel="Altitude [m]",ylabel="R²",grid=True,legend=True),
                           height_levels:np.ndarray=get_height_level_range(1680,16980,step=60),
                           ):
    """
    ### todo: return individual height plots as figures, when I did it now, it always plotted the last plot only ###
    
    returns:
        - hex plot for all data
        - list: hex plts for different height levels
        - hvplt: metrics per heigh levls plt
        - pd.DataFrame: metrics per heigh levls
    """                 
    dardar_cloud_cover_data_1d = torch.flatten(dardar_cloud_cover_data)
    y_hat_cloud_cover_data_1d = torch.flatten(y_hat_cloud_cover_data)          
    
    # create hex plt for all data
    all_data_plt = horizontal_cloud_cover_plt(dardar_cloud_cover_data_1d, y_hat_cloud_cover_data_1d,plt_kwargs=hex_plt_kwargs)   
                           
    # create plt for height levels
    r2_levels = []
    corr_levels = []
    # height_plts = []
    for height_bin in range(dardar_cloud_cover_data.shape[1]-1):
        r2_levels.append(r2_score(dardar_cloud_cover_data[:,height_bin].numpy(),y_hat_cloud_cover_data[:,height_bin].numpy()))
        corr_levels.append(np.corrcoef(dardar_cloud_cover_data[:,height_bin],y_hat_cloud_cover_data[:,height_bin])[0][1])
        # create one plot for each cumulated height level (16 levels)
        # height_plts.append(horizontal_cloud_cover_plt(dardar_cloud_cover_data[:,height_bin], y_hat_cloud_cover_data[:,height_bin],title=f"Horizontal cloud cover ({height_bins[height_bin*16]}m - {height_bins[(height_bin+1)*16]}m)"))
                           
    # create summary statistics plt
    n_level_aggregation = int(height_levels.shape[0] / dardar_cloud_cover_data.shape[1])

    hor_cloud_cover_metrics = pd.DataFrame(index=[h for i,h in enumerate(height_levels) if i % n_level_aggregation ==0][:-4]) #[:-1]
    hor_cloud_cover_metrics["R2"] = r2_levels
    hor_cloud_cover_metrics["corr"] = corr_levels

    summary_line_plt = hor_cloud_cover_metrics["R2"][hor_cloud_cover_metrics.index>4000].hvplot.line(**line_plt_kwargs)
                           
    return all_data_plt, summary_line_plt, hor_cloud_cover_metrics # , height_plts

def horizontal_cloud_cover_plt(dardar_cloud_cover_data_1d, y_hat_cloud_cover_data_1d, plt_kwargs=dict(kind="hex",joint_kws={'gridsize':40, 'bins':'log',"mincnt":100,"cmap":plt.get_cmap("Blues",8)}),title="Cloud cover frequency", regplt=True):
    sns.set(font_scale=1.65)
    sns.set_style("white")
    g = sns.jointplot(x=dardar_cloud_cover_data_1d,y=y_hat_cloud_cover_data_1d, **plt_kwargs)
    g.fig.axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, p: f'{y:.0%}'))
    g.fig.axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.0%}'))
    # g.fig.axes[0].set_xticklabels([f'{x:.0%}' for x in xticks])#
    r2 = r2_score(dardar_cloud_cover_data_1d.numpy(),y_hat_cloud_cover_data_1d.numpy())
    mae = mean_absolute_error(dardar_cloud_cover_data_1d, y_hat_cloud_cover_data_1d)
    
    sns.lineplot(x=[0,1],y=[0,1],linestyle="--",color="grey")
    g.fig.text(0.2, 0.7, 'R² = ' + str(round(r2,2)),fontsize=20)
    g.fig.text(0.2, 0.6, 'MAE = ' + str(round(mae,2)),fontsize=20)
    if regplt:
        reg = sns.regplot(x=y_hat_cloud_cover_data_1d.numpy(),y=dardar_cloud_cover_data_1d.numpy(),ax=g.ax_joint,scatter=False,color="red")

        #calculate slope and intercept of regression equation
        slope, intercept, r, p, sterr = scipy.stats.linregress(x=reg.get_lines()[0].get_xdata(),
                                                               y=reg.get_lines()[0].get_ydata())
     
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # shrink fig so cbar is visible
    cbar_ax = g.fig.add_axes([.7, .15, .03, .4])  # x, y, width, height
    plt.colorbar(cax=cbar_ax,label="# DARDAR overpasses")
    cbar_ax.yaxis.set_label_position('left')
    g.ax_marg_y.set_xscale('log')
    g.ax_marg_x.set_yscale('log')
    g.set_axis_labels(ylabel="IceCloudNet cloud coverage",xlabel="DARDAR cloud coverage")
    g.fig.suptitle(f"{title}")
    
    return g

def calc_nan_mean(y):
    """in cloud mean"""
    masked_profile = y.masked_fill(y==0, torch.nan)
    mean_profile = masked_profile.nanmean(dim=0).unsqueeze(0)
    return mean_profile

def calc_all_cloud_mean(y):
    """grid mean"""
    return y.mean(dim=0).unsqueeze(0)

def get_lat_mean(profile_list: list[torch.tensor], lat_list: torch.tensor, height_level_range: np.ndarray=get_height_level_range(),target_transform=LogTransform(scaler=1e7),mean_type="grid"):
    """calculate in cloud zonal mean (1°) per height level (60m)"""
    
    assert mean_type in ["grid","incloud"]
    
    mean_func = calc_all_cloud_mean if mean_type=="grid" else calc_nan_mean
    
    # for each profile calculate in cloud mean per height level
    # mean_profiles = [calc_nan_mean(y) for y in profile_list] # in cloud mean
    if target_transform is not None:
        mean_profiles = [mean_func(target_transform.inverse_transform(y)) for y in profile_list]
    else:
        mean_profiles = [mean_func(y) for y in profile_list]
        
    mean_profiles = torch.concat(mean_profiles,0)
    
    # create df 
    df = pd.DataFrame(mean_profiles,columns=height_level_range)
    df["lat"]=torch.round(lat_list)#,decimals=0) # round lat to one degree
    
    # aggregate in cloud per lat
    df_lat_mean = df.groupby("lat").agg(np.nanmean)
        
    return df, df_lat_mean

def get_lat_count(profile_list: list[torch.tensor], lat_list: torch.tensor, height_level_range: np.ndarray=get_height_level_range()):
    """calculate number of in cloud data per 1° lat and height level (60m)"""
    
    # for each profile calculate in cloud count
    in_cloud_count_profiles = [torch.sum(y>0,0).unsqueeze(0) for y in profile_list]
    in_cloud_count_profiles = torch.concat(in_cloud_count_profiles,0)
    
    # create dataframe and aggregate over lats
    df_count = pd.DataFrame(in_cloud_count_profiles,columns=height_level_range)
    df_count["lat"]=torch.round(lat_list)#,decimals=0)
    df_count_lat_mean = df_count.groupby("lat").sum()
    
    return df_count, df_count_lat_mean

def zonal_mean(y_hat_profile_list, 
               dardar_profile_list, 
               lat_list, 
               target_variable="iwc",
               mean_type="grid",
               target_transform=LogTransform(scaler=1e7),
               height_levels:np.ndarray=get_height_level_range(1680,16980,step=60),
               min_count:int=500,
               contour_kwargs=dict(locator=ticker.LogLocator(),cmap="Blues",levels=np.arange(4000,18000,2000)),
               diff_contour_kwargs=dict(locator=ticker.LinearLocator(),cmap="bwr",norm=Normalize(-2,2),levels=np.arange(-2,2.25,0.25)),
               selected_height_levels=None,
               fig=None,
               ax=None):

    assert mean_type in ["grid","incloud"]
    
    if len(y_hat_profile_list[0].shape) == 3:
        # multi pred → select one variable only
        var_idx = 0 if target_variable == "iwc" else 1
        y_hat_profile_list = [p[:,var_idx] for p in y_hat_profile_list]
        dardar_profile_list = [p[:,var_idx] for p in dardar_profile_list]
    
    #  select height levels in profile list
    if selected_height_levels is not None:
        y_hat_profile_list = [p[:,selected_height_levels] for p in y_hat_profile_list]
        dardar_profile_list = [p[:,selected_height_levels] for p in dardar_profile_list]
                
    contour_kwargs["norm"]=colors.LogNorm(*plt_kwargs_commons["var_range"][target_variable])
    contour_kwargs["levels"] = np.logspace(*plt_kwargs_commons["var_range_log"][target_variable],13)
    cticks = np.logspace(*plt_kwargs_commons["var_range_log"][target_variable],plt_kwargs_commons["var_range_log"][target_variable][1]-plt_kwargs_commons["var_range_log"][target_variable][0]+1) # tick at full oom
    var_label = plt_kwargs_commons["axis_title"][target_variable]
    
    if target_variable=="iwc":
        fig_idxs = [0,2,4] 
    else:
        fig_idxs = [1,3,5]

    y_hat_df, y_hat_df_lat_mean = get_lat_mean(y_hat_profile_list, lat_list,height_level_range=height_levels,mean_type=mean_type,target_transform=target_transform)
    dardar_df, dardar_df_lat_mean = get_lat_mean(dardar_profile_list, lat_list,height_level_range=height_levels,mean_type=mean_type,target_transform=target_transform)

    y_hat_df_count,  y_hat_df_lat_count   = get_lat_count(y_hat_profile_list, lat_list,height_level_range=height_levels)
    dardar_df_count, dardar_df_lat_count = get_lat_count(dardar_profile_list, lat_list,height_level_range=height_levels)
    
    # prepare data for plotting
    X = y_hat_df_lat_mean.index.values
    Y = y_hat_df_lat_mean.columns
    
    # y_hat
    y_hat_Z = y_hat_df_lat_mean.values.T
    # y_hat_Z = dm.target_transform.inverse_transform(y_hat_Z)
    y_hat_Z_count = y_hat_df_lat_count.values.T.astype(np.float32)
    y_hat_mask = y_hat_Z_count<min_count # mask points with less realisations than min count

    # dardar
    dardar_Z = dardar_df_lat_mean.values.T
    # dardar_Z = dm.target_transform.inverse_transform(dardar_Z)
    dardar_Z_count = dardar_df_lat_count.values.T.astype(np.float32)
    dardar_mask = dardar_Z_count<min_count # mask points with less realisations than min count
    
    # zonal mean plots
    if fig is None:
        fig, ax = plt.subplots(3, 2, figsize=(16, 16))
        ax = ax.flatten()
    
        for label, ax_ in zip(['A', 'B', 'C', 'D', 'E', 'F'],ax):
            trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
            ax_.text(0.0, 1.0, label, transform=ax_.transAxes + trans,
                    fontsize=30, va='bottom', fontfamily='sans-serif',fontweight="bold")  
    
    # dardar
    contour1 = ax[fig_idxs[0]].contourf(X, Y, np.ma.array(dardar_Z,mask=dardar_mask,fill_value=np.nan),**contour_kwargs)
    ax[fig_idxs[0]].set_title('DARDAR Zonal Mean' )
    ax[fig_idxs[0]].set_xlabel('Latitude')
    ax[fig_idxs[0]].set_ylabel('Altitude [m]')
    cbar = plt.colorbar(contour1, format='%.e', ticks=cticks, label=var_label,ax=ax[fig_idxs[0]])
    
    # predictions
    contour2 = ax[fig_idxs[1]].contourf(X, Y, np.ma.array(y_hat_Z,mask=y_hat_mask,fill_value=np.nan),**contour_kwargs)
    ax[fig_idxs[1]].set_title('IceCloudNet Zonal Mean')
    ax[fig_idxs[1]].set_xlabel('Latitude')
    ax[fig_idxs[1]].set_ylabel('Altitude [m]')
    cbar = plt.colorbar(contour2, format='%.e', ticks=cticks, label=var_label,ax=ax[fig_idxs[1]])
    
    # diff plot Dardar - predictions
    diff_z_log = np.log10(y_hat_Z)-np.log10(dardar_Z)
    diff_z_log_masked = np.ma.array(diff_z_log, mask=(np.isfinite(diff_z_log)==False) | dardar_mask,fill_value=np.nan) # mask if values is not finite (cause of the log trafo) or not enough datapoints are available
    
    #levels = np.array([-1e-4,-9e-5,-8e-5,-6e-5,-5e-5,-4e-5,-2e-5,-1e-5,-5e-6,-1e-6,0,1e-6,5e-6,1e-5,2e-5])
    #contour3=ax[4].contourf(X,Y,diff_z,**dict(cmap="seismic",levels=levels,locator=ticker.FixedLocator(levels),norm=colors.CenteredNorm()))
    #cbar = plt.colorbar(contour3, label="IWC [kg m⁻³]", format='%.e',ax=ax[4])
    #contour3.axes.set_title("Predictions-Dardar")
    
    cbar_ticks = [level for i,level in enumerate(diff_contour_kwargs["levels"]) if i % 2 == 0]
    contour3 = ax[fig_idxs[2]].contourf(X, Y, diff_z_log_masked ,**diff_contour_kwargs) 
    cbar = plt.colorbar(contour3, format='%.2f', ticks=diff_contour_kwargs["locator"], label=f"Log10 {var_label}",ax=ax[fig_idxs[2]])
    cbar.set_ticks(cbar_ticks)

    ax[fig_idxs[2]].set_title('Zonal Mean | IceCloudNet - Dardar')
    ax[fig_idxs[2]].set_xlabel('Latitude')
    ax[fig_idxs[2]].set_ylabel('Altitude [m]')

    plt.tight_layout()
    
    # zonal mean - count plots
#     fig_count, ax = plt.subplots(1, 2, figsize=(14, 5))

#     contour1 = ax[0].contourf(X, Y, np.ma.array(y_hat_Z_count,mask=y_hat_mask,fill_value=np.nan))
#     ax[0].set_title('Zonal Mean predictions - Pixel count')
#     ax[0].set_xlabel('Latitude')
#     ax[0].set_ylabel('height [m]')
#     cbar = plt.colorbar(contour1,label="pixels",ax=ax[0])

#     contour2 = ax[1].contourf(X, Y, np.ma.array(dardar_Z_count,mask=dardar_mask,fill_value=np.nan))
#     ax[1].set_title('Zonal Mean - Dardar - Pixel count')
#     ax[1].set_xlabel('Latitude')
#     ax[1].set_ylabel('height [m]')
#     cbar = plt.colorbar(contour2, label="pixels",ax=ax[1])

#     plt.tight_layout()
    
    
    return fig #, X,Y,dardar_Z, y_hat_Z #, fig_count
    
def calc_all_cloud_iwp_mean(y,level_thickness=60):
    """
        args:
            y (torch.tensor): 2d iwc profile tensor in original scale (n overpass pixels, n height levels)
            level_thickness (int)
        returns:
            (torch.tensor): iwp mean of profile
    """
    iwp = y.sum(dim=1)*level_thickness
    iwp_mean = torch.mean(iwp).unsqueeze(0)
    
    return iwp_mean

def calc_nan_profile_mean(y):
    """in cloud mean for whole profile"""
    masked_profile = y.masked_fill(y==0, torch.nan)
    mean = torch.nanmean(masked_profile)
    return mean

def calc_in_cloud_nice_mean(y,height_levels,selected_levels=(12000,10000)):
    selected_heigt_levels = np.where((height_levels<=selected_levels[0]) & (height_levels>=selected_levels[1]))[0]
    # print("selected height levels", selected_heigt_levels)
    mean = calc_nan_profile_mean(y[:,selected_heigt_levels]).unsqueeze(0)
    return mean

def get_regional_mean(profile_list, lat_list, lon_list, target_transform=LogTransform(scaler=1e7), grid_size=5, level_thickness=60,height_levels=get_height_level_range(),target_variable="iwc"):
    if target_variable == "iwc":
        mean_profiles = [calc_all_cloud_iwp_mean(target_transform.inverse_transform(y),level_thickness=level_thickness) for y in profile_list]
    elif target_variable == "nice":
        mean_profiles = [calc_in_cloud_nice_mean(target_transform.inverse_transform(y),height_levels) for y in profile_list]
    mean_profiles = torch.concat(mean_profiles,0)
    
    df = pd.DataFrame(mean_profiles,columns=["IWP"])
    df["lat"]=torch.round(lat_list,decimals=0) # round lat to one degree

    df["lon"]=torch.round(lon_list,decimals=0) # round lat to one degree

    df["lon"] = df["lon"].apply(lambda x: custom_round(x,base=grid_size))
    df["lat"] = df["lat"].apply(lambda x: custom_round(x,base=grid_size))

    df_agg = df.groupby(["lat","lon"]).mean().unstack()["IWP"]
    
    X=df_agg.columns
    Y=df_agg.index.values
    Z=df_agg.values
    
    return df_agg,X,Y,Z
        

def custom_round(x, base=1):
    return int(base * round(float(x)/base))

def iwp_regional_mean(y_hat_profile_list, 
                      dardar_profile_list, 
                      lat_list, 
                      lon_list,
                      target_variable="iwc",
                      target_transform=LogTransform(scaler=1e7),
                      level_thickness=60,
                      height_levels=get_height_level_range(),
                      grid_size=4,
                      contour_kwargs=dict(norm=colors.LogNorm(vmin=1e-4, vmax=1e0),cmap="viridis"), # levels = np.logspace(-4,0,9),locator=ticker.LogLocator()
                      diff_contour_kwargs=dict(cmap="bwr",norm=Normalize(-1,1)),
                      fig=None,
                      axs=None):
    
    # multi pred → select one variable only
    if len(y_hat_profile_list[0].shape) == 3:
        var_idx = 0 if target_variable == "iwc" else 1
        y_hat_profile_list = [p[:,var_idx,:] for p in y_hat_profile_list]
        dardar_profile_list = [p[:,var_idx,:] for p in dardar_profile_list]
    
    y_hat_df, y_hat_X, y_hat_Y, y_hat_Z = get_regional_mean(y_hat_profile_list, lat_list,lon_list,level_thickness=level_thickness,grid_size=grid_size,target_variable=target_variable,target_transform=target_transform,height_levels=height_levels)
    dardar_df, dardar_X, dardar_Y, dardar_Z = get_regional_mean(dardar_profile_list, lat_list,lon_list,level_thickness=level_thickness,grid_size=grid_size,target_variable=target_variable,target_transform=target_transform,height_levels=height_levels)
    
    if target_variable=="iwc":
        clabel = "IWP [kg m⁻²]"
        plt_title = "IWP"
        cticks = np.logspace(-4,0,5)
    elif target_variable == "nice":
        clabel = "Nice [m⁻³]"
        plt_title = "N$_{ice}$ (10-12 km)"
        cticks = [1e4,2e4,5e4,1e5,2e5]
    
    
    if target_variable=="iwc":
        fig_idxs = [0,2,4] 
    else:
        fig_idxs = [1,3,5]
    
    if fig is None:  
        fig, axs = plt.subplots(3, 2, figsize=(16, 16))
        axs = axs.flatten()
    
    for label, ax in zip(['A', 'B', 'C', 'D', 'E', 'F'],axs):
        trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize=30, va='bottom', fontfamily='sans-serif',fontweight="bold")  
    
    contour1=axs[fig_idxs[0]].pcolormesh(dardar_X,dardar_Y,dardar_Z,**contour_kwargs)
    axs[fig_idxs[0]].set_title(f'Dardar {plt_title}')
    axs[fig_idxs[0]].set_xticks(np.linspace(-30,30,7))
    axs[fig_idxs[0]].set_yticks(np.linspace(-30,30,7))
    axs[fig_idxs[0]].set_ylabel('Latitude')
    axs[fig_idxs[0]].set_xlabel('Longitude')
    cbar = plt.colorbar(contour1,format='%.e', ticks=cticks,label=clabel,ax=axs[fig_idxs[0]])#,ticks=np.logspace#,ticks=contour_kwargs["locator"])
    axs[fig_idxs[0]].coastlines(linewidth=5,color="white")
    # cbar.ax.set_clim(vmin=1e-4, vmax=5e-1)
    
    
    
    contour2=axs[fig_idxs[1]].pcolormesh(y_hat_X,y_hat_Y,y_hat_Z,**contour_kwargs)
    axs[fig_idxs[1]].coastlines(linewidth=5,color="white")
    axs[fig_idxs[1]].set_title(f'IceCloudNet {plt_title}')
    axs[fig_idxs[1]].set_xticks(np.linspace(-30,30,7))
    axs[fig_idxs[1]].set_yticks(np.linspace(-30,30,7))
    axs[fig_idxs[1]].set_ylabel('Latitude')
    axs[fig_idxs[1]].set_xlabel('Longitude')
    cbar = plt.colorbar(contour2,format='%.e', ticks=cticks, label=clabel,ax=axs[fig_idxs[1]])#,ticks=contour_kwargs["locator"]
    # cbar.ax.set_clim(vmin=1e-4, vmax=5e-1)
    
    # diff plot
    # diff_z_log = np.log10(y_hat_Z)-np.log10(dardar_Z) 
    # diff_z_log_masked = np.ma.array(diff_z_log, mask=(np.isfinite(diff_z_log)==False),fill_value=np.nan) # mask if values is not finite (cause of the log trafo) or not enough datapoints are available
    # diff_z_log = log_diff_exp(dardar_Z,y_hat_Z)
    diff_z_log_masked = np.log10(y_hat_Z/dardar_Z)
    # cbar_ticks = [level for i,level in enumerate(diff_contour_kwargs["levels"]) if i % 4 == 0]
    contour3 = axs[fig_idxs[2]].pcolormesh(dardar_X, dardar_Y, diff_z_log_masked,**diff_contour_kwargs) 
    axs[fig_idxs[2]].coastlines(linewidth=5,color="black")
    cbar = plt.colorbar(contour3, format='%.1f',label=f"Log10 {clabel}",ax=axs[fig_idxs[2]],ticks=np.arange(-1,1.25,0.25))
    # cbar.set_ticks(cbar_ticks)
    
    axs[fig_idxs[2]].set_title(f'IceCloudNet - Dardar')
    axs[fig_idxs[2]].set_xticks(np.linspace(-30,30,7))
    axs[fig_idxs[2]].set_yticks(np.linspace(-30,30,7))
    axs[fig_idxs[2]].set_ylabel('Latitude')
    axs[fig_idxs[2]].set_xlabel('Longitude')
    
    plt.tight_layout()
    return fig, dardar_Z, y_hat_Z

