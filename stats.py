from utils_2 import eval_grab, load_masters
import eval_utils as nets

import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pingouin as pg

import numpy as np
import operator
import random

from all_net_eval import visualise_s2n3, visualise_s1, perm_test

def grab_comp(df, study, nway=2, metric='LPIPS', subject=None, roi='VC', vox='1pt8mm'):

    if subject is None:
        subject = [1, 2, 3, 4, 5, 6, 7, 8]
    comp =(df[
        (df['study'] == study) & (df['nway'] == nway) & (df['metric'] == metric) & (df['subject'].isin(subject))
        & (df['ROI'] == roi) & (df['vox_res'] == vox)
               ])

    return comp


def grab_scores(df, study, nway=2, metric='LPIPS', roi='VC', vox='1pt8mm'):

    comp =(df[
        (df['study'] == study) & (df['nway'] == nway) & (df['metric'] == metric)
        & (df['ROI'] == roi) & (df['vox_res'] == vox)
               ])

    scores = np.array(comp['score'][0])

    return scores


def study2_analysis(df, dataframe, nways):
    base_p = 0.05

    # Study 2
    # Parameters
    study = 2
    # Testing voxel_resolution
    corrections = 6
    corrected_p = base_p / corrections
    # effect = 'voxel resolution'
    metrics = ['LPIPS', 'PCC']
    comparison = 'voxel resolution'
    comp_labels = ['1pt8mm', '3mm']

    for metric in metrics:
        for n in nways:
            # 1.8mm
            comp_1 = grab_scores(df, study=study, vox=comp_labels[0], metric=metric, nway=n)
            mean_1 = np.mean(comp_1)
            std_1 = np.std(comp_1)

            # 3mm
            comp_2 = grab_scores(df, study=study, vox=comp_labels[1], metric=metric, nway=n)
            mean_2 = np.mean(comp_2)
            std_2 = np.std(comp_2)

            # t, p = stats.wilcoxon(comp_1, comp_2)

            wilcox = pg.wilcoxon(comp_1, comp_2, alternative='two-sided', correction='false')
            print(wilcox)

            CI = non_param_paired_CI(comp_1, comp_2, 0.95)
            # TODO: redefine and add new CI
            print(CI)

            p = wilcox.iloc[0]['p-val']
            t = wilcox.iloc[0]['W-val']
            sig = p < corrected_p

            dataframe['comparison'].append(comparison)
            dataframe['comp_1_label'].append(comp_labels[0])
            dataframe['comp_2_label'].append(comp_labels[1])
            dataframe['comp_1_scores'].append(comp_1)  # TODO: Check this is getting saved properly
            dataframe['comp_2_scores'].append(comp_2)
            dataframe['comp_diffs'].append(comp_1 - comp_2)
            dataframe['metric'].append(metric)
            dataframe['nway'].append(n)
            dataframe['comp_1_mean'].append(mean_1)
            dataframe['comp_1_mdn'].append(np.median(comp_1))
            dataframe['comp_1_std'].append(std_1)
            dataframe['comp_2_mean'].append(mean_2)
            dataframe['comp_2_mdn'].append(np.median(comp_2))
            dataframe['comp_2_std'].append(std_2)
            dataframe['mean_diff'].append(np.mean(comp_1 - comp_2))
            dataframe['mdn_diff'].append(np.median(comp_1 - comp_2))
            dataframe['t'].append(t)
            dataframe['p'].append(p)
            dataframe['CI'].append(CI)
            dataframe['RBC'].append(wilcox.iloc[0]['RBC'])
            dataframe['CLES'].append(wilcox.iloc[0]['CLES'])
            # TODO CI and effect

            if sig:
                if p < 0.001:
                    p_report = '.001'
                wilcoxon_statement = 'A two-tailed Wilcoxon signed-rank test was conducted to evaluate the effects of {}' \
                                     ' on reconstruction accuracy. A Bonferroni correction applied to account for multiple tests across ' \
                                     'both metrics, resulting in a significance level of {:.3f}. Results indicated that models trained on ' \
                                     '1.8mm voxel resolution achieved greater reconstruction accuracy scores measured using the {} {}-way task ' \
                                     '(M = {:.1f}, SD = {:.1f}) than did those trained on 3mm voxels (M = {:.1f}, SD = {:.1f}), t = {:.2f}, p < {}. '.format(
                    comparison, corrected_p, metric, n, mean_1, std_1, mean_2, std_2, t, p_report)
            else:
                p_report = '{:.3f}'.format(p)
                p_report = p_report.replace('0.','.')
                wilcoxon_statement = 'A two-tailed Wilcoxon signed-rank test was conducted to evaluate the effects of {}' \
                                     ' on reconstruction accuracy. A Bonferroni correction applied to account for multiple tests across ' \
                                     'both metrics, resulting in a significance level of {:.3f}. Results indicated that there was no significant difference in reconstruction accuracy ({} {}-way) scores between models trained on ' \
                                     '1.8mm voxel resolution (M = {:.1f}, SD = {:.1f}) and those trained on 3mm voxels (M = {:.1f}, SD = {:.1f}), t = {:.2f}, p = {:.3f}.'.format(
                    comparison, corrected_p, metric, n, mean_1, std_1, mean_2, std_2, t, p_report)
            print(wilcoxon_statement)

    return dataframe, wilcox, CI


def plot_hist(data, label, type='scores', plot_ci=False, ci=None):
    plt.cla()
    kwargs = dict(alpha=0.2, bins=50)

    colours = ['b','r','g']
    count = 0
    for d in data:
        colour = colours[count]
        plt.hist(d, **kwargs, color=colour, label=label[count])
        plt.axvline(d.mean(), color='k', linestyle='dashed', linewidth=1)
        count += 1

    if type=='scores':
        plt.gca().set(title='Distribution of Accuracy Scores', ylabel='Frequency')
    else:
        plt.gca().set(title='Distribution of Accuracy Differences Scores', ylabel='Frequency')

    if plot_ci and ci is not None:
        plt.axvline(ci[0], color='r', linestyle='dashed', linewidth=1)
        plt.axvline(ci[1], color='b', linestyle='dashed', linewidth=1)

    plt.legend()
    plt.show()

    # custom func for confidence interval for differences
    # for (non-Gaussian paired data) Example: Wilcoxon signed-rank test
    # https://towardsdatascience.com/prepare-dinner-save-the-day-by-calculating-confidence-interval-of-non-parametric-statistical-29d031d079d0

def non_param_paired_CI(sample1, sample2, conf):
    n = len(sample1)
    alpha = 1-conf
    N = stats.norm.ppf(1 - alpha/2)

    # The confidence interval for the difference between the two population
    # medians is derived through the n(n+1)/2 possible averaged differences.
    diff_sample = sorted(list(map(operator.sub, sample2, sample1)))
    averages = sorted([(s1+s2)/2 for i, s1 in enumerate(diff_sample) for _, s2 in enumerate(diff_sample[i:])])

    # the Kth smallest to the Kth largest of the averaged differences then
    # determine the confidence interval, where K is:
    k = np.math.ceil(n*(n+1)/4 - (N * (n*(n+1)*(2*n+1)/24)**0.5))

    CI = (round(averages[k-1],3), round(averages[len(averages)-k],3))
    return CI


def bootstrap_acc(scores):
    boot_orig = scores
    boot_orig = np.array(boot_orig)
    # mean_orig = np.mean(boot_orig)

    bootstrapped_means = []

    for i in range(10000):
        # TODO: Check how many the sample
      y = random.sample(boot_orig.tolist(), 1000)
      avg = np.mean(y)
      bootstrapped_means.append(avg)

    bootstrapped_means_all = np.mean(bootstrapped_means)

    bootstrapped_means = np.array(bootstrapped_means)

    lower_bound = np.percentile(bootstrapped_means, 2.5)
    upper_bound = np.percentile(bootstrapped_means, 97.5)
    ci = [lower_bound, upper_bound]

    # plot_hist([bootstrapped_means], 'bootstrap dist', plot_ci=True, ci=ci)

    # z_lower = (ci[0]-bootstrapped_means_all)/np.std(bootstrapped_means)
    # p, sig = perm_test(bootstrapped_means, boot_orig)
    return ci


full_df = pd.read_pickle('D:/Lucha_Data/final_networks/output/full_dataset.pkl')
data_df = pd.read_pickle('D:/Lucha_Data/final_networks/output/full_data_study_2n3.pkl')

data = dict(comparison=[],comp_1_label=[],comp_2_label=[],comp_1_scores=[],comp_2_scores=[],comp_diffs=[],metric=[],
            nway=[],comp_1_mean=[],comp_1_mdn=[],comp_1_std=[],comp_2_mean=[],comp_2_mdn=[],comp_2_std=[], mean_diff=[],
            mdn_diff=[],t=[],p=[],CI=[],RBC=[],CLES=[])
# TODO: Add CI and effect size ,ci=[],effect_size=[]

# Study 2
nways=[2,5,10]
data_comp, wilcox, CI = study2_analysis(data_df, data, nways)

data_comp_df = pd.DataFrame.from_dict(data_comp)

# wilcox.iloc[0]['W-val']


"""plot_hist([data_comp_df.iloc[0]['comp_1_scores']], label=['LPIP 1.8mm 2-way'], type='scores')
plot_hist([data_comp_df.iloc[0]['comp_2_scores']], label=['LPIP 3mm 2-way'], type='scores')
# plot_hist([data_comp_df.iloc[0]['comp_1_scores'],data_comp_df.iloc[0]['comp_2_scores']], label=['LPIP 1.8mm 2-way', 'LPIP 3mm 2-way'], type='scores')
plot_hist([data_comp_df.iloc[0]['comp_diffs']], label=['LPIP 1.8 vs 3 diff 2-way'], type='scores')

plot_hist([data_comp_df.iloc[1]['comp_1_scores']], label=['LPIP 1.8mm 5-way'], type='scores')
plot_hist([data_comp_df.iloc[1]['comp_2_scores']], label=['LPIP 3mm 5-way'], type='scores')
# plot_hist([data_comp_df.iloc[0]['comp_1_scores'],data_comp_df.iloc[0]['comp_2_scores']], label=['LPIP 1.8mm 2-way', 'LPIP 3mm 2-way'], type='scores')
plot_hist([data_comp_df.iloc[1]['comp_diffs']], label=['LPIP 1.8 vs 3 diff 5-way'], type='scores')

plot_hist([data_comp_df.iloc[2]['comp_1_scores']], label=['LPIP 1.8mm 10-way'], type='scores')
plot_hist([data_comp_df.iloc[2]['comp_2_scores']], label=['LPIP 3mm 10-way'], type='scores')
# plot_hist([data_comp_df.iloc[0]['comp_1_scores'],data_comp_df.iloc[0]['comp_2_scores']], label=['LPIP 1.8mm 2-way', 'LPIP 3mm 2-way'], type='scores')
plot_hist([data_comp_df.iloc[2]['comp_diffs']], label=['LPIP 1.8 vs 3 diff 10-way'], type='scores')"""
# stats.wilcoxon(comp_1['lpips'], comp_2['lpips'])
# TEST
# df = pd.read_pickle('D:/Lucha_Data/final_networks/output/full_data_study_2n3.pkl')
# study = 2
# metric = 'LPIPS'
# n = 2

# spss_grab = grab_comp(data_df,2,2,'LPIPS',roi='VC',vox='3mm')
# spss_test = data_comp_df.iloc[0].explode('')


