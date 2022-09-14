import time
import seaborn as sns
import itertools
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats

import numpy as np
import operator
import random

from scipy.stats import norm

from utils_2 import permutation

def non_param_paired_CI(sample1, sample2, conf):
    # custom func for confidence interval for differences
    # for (non-Gaussian paired data) Example: Wilcoxon signed-rank test
    # https://towardsdatascience.com/prepare-dinner-save-the-day-by-calculating-confidence-interval-of-non-parametric-statistical-29d031d079d0

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


def bootstrap_acc(scores, permutations=10000, observations=1000):
    boot_orig = scores
    boot_orig = np.array(boot_orig)
    mean_orig = np.mean(boot_orig)

    bootstrapped_means = []

    for i in range(permutations):
        # TODO: Check how many the sample
      y = random.sample(boot_orig.tolist(), observations)
      avg = np.mean(y)
      bootstrapped_means.append(avg)

    bootstrapped_means_all = np.mean(bootstrapped_means)

    bootstrapped_means = np.array(bootstrapped_means)

    lower_bound = np.percentile(bootstrapped_means, 2.5)
    upper_bound = np.percentile(bootstrapped_means, 97.5)
    ci = [lower_bound, upper_bound]

    upper_diff = upper_bound - mean_orig
    lower_diff = mean_orig - lower_bound
    ci_diff = [lower_diff, upper_diff]

    # plot_hist([bootstrapped_means], 'bootstrap dist', plot_ci=True, ci=ci)

    # z_lower = (ci[0]-bootstrapped_means_all)/np.std(bootstrapped_means)
    # p, sig = perm_test(bootstrapped_means, boot_orig)
    return ci, ci_diff


def permute(repeats=100):
    # TODO: Change this for laptop
    root_dir = 'D:/Lucha_Data/final_networks/output/'
    # root_dir = 'C:/Users/david/Documents/Thesis/final_networks/output/'  # FOR laptop
    networks = os.path.join(root_dir,'all_eval/')
    save_path = root_dir

    nway = True
    pairwise = False

    count = 0

    # Set n way comparisons
    ns = [2, 5, 10]
    # 1000 according to Guy Gaziv
    repeats = repeats

    # Add new evaluation requests
    data = {
        **{'run_name': [], 'score': [], 'study': [], 'subject': [], 'vox_res': [], 'ROI': [], 'set_size': [],
           'metric': [], 'nway': [], 'repeats': []}
    }

    for folder in os.listdir(networks):
        start = time.time()
        count += 1

        # print(folder)
        folder_dir = os.path.join(networks, folder)

        # Get name of network
        sep = '_Stage3_'
        network = folder.split(sep, 1)[0]  # aka run_name

        study, subj, vox_res, ROI, set_size = str(network).split('_')
        study = int(study.split('y', 1)[1])
        subj = int(subj.split('0', 1)[1])
        eval_options = dict(study=study, subject=subj, vox_res=vox_res, ROI=ROI, set_size=set_size, repeats=repeats)

        print('Evaluating network: {}'.format(network))

        # Load data
        print('Reading Data')
        pcc = pd.read_csv(os.path.join(folder_dir, 'pcc_table.csv'), index_col=0)
        # ssim = pd.read_excel(os.path.join(folder_dir, 'ssim_table.xlsx'), engine='openpyxl', index_col=0)
        lpips = pd.read_csv(os.path.join(folder_dir, 'lpips_table.csv'), index_col=0)

        # I know this is redundant but...
        pcc = pcc.to_numpy()
        lpips = lpips.to_numpy()
        print('Data Ready')

        if nway:
            print('Running n-way comparisons')
            for n in ns:
                way_label = '{}-way'.format(n)

                print('Running {} comparisons'.format(way_label))
                pcc_nway_out = permutation(pcc, n=n, repeats=repeats, metric="pcc")
                print('PCC Complete')

                # ssim_nway_out = nway_comp(ssim, n=n, repeats=repeats, metric="ssim")
                # print('SSIM Complete')
                lpips_nway_out = permutation(lpips, n=n, repeats=repeats, metric="lpips")
                print('LPIPS Complete')

                scores = [pcc_nway_out, lpips_nway_out]
                # Just to have an if statement for PCC and LPIPS
                score_check = 0

                for score in scores:
                    score_check += 1
                    for header, value in eval_options.items():
                        data[header].append(value)

                    data['run_name'].append(network)
                    data['score'].append(score)
                    data['nway'].append(n)
                    if score_check == 1:
                        data['metric'].append('PCC')
                    else:
                        data['metric'].append('LPIPS')

                print('Evaluations saved to master list.')

        end = time.time()
        print('Time per run =', end - start)
        # stops it after study 1 studies
        if count == 8:
            break

        # if count == 8:
        #     # raise Exception('check masters')
        #     break

    print('Complete.')
    df = pd.DataFrame.from_dict(data)
    # df.to_pickle(os.path.join(save_path, 'full_dataset.pkl'))
    return df


def show_values(axs, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = (p.get_y() + p.get_height() + (p.get_height()*0.01))/2
                value = '{:.3f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center")
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.3f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)


def visualise_s1(df):
    plt.cla()
    chained = lambda l: list(itertools.chain(*l))
    sns.set()
    sns.set_context("paper", font_scale=1.4)

    target_exps = chained([
                              f'Study1_SUBJ0{sbj_num}_1pt8mm_VC_max',
                          ] for sbj_num in [1, 2, 3, 4, 5, 6, 7, 8])

    my_df = df[(df.nway > 0) & (df.nway <= 1000) &
               (df.run_name.isin(target_exps)) & (df.metric == 'LPIPS')
               ]

    # fn = lambda x: '_'.join(x.split('_')[1:])
    # my_df.exp_name = my_df.exp_name.apply(fn)
    my_df = my_df.explode('score')
    # my_df.exp_name.unique()

    # g = sns.catplot(x='nway', y='score', hue='run_name', col='subject', kind="bar", data=my_df, legend=False,
    #                 legend_out=True, sharey=False)
    g = sns.barplot(data=my_df, x="nway", y='score', hue='subject')
    # g.set_titles(col_template='Subject')
    # graph = my_df.groupby(['nway','subject']).plot(kind='bar')

    # plt.hlines(y=0.5, xmin='0', xmax='2')
    # plt.hlines(y=0.2, xmin=5, xmax=5)

    plt.gcf().set_size_inches(8, 4)
    plt.gcf().tight_layout(rect=[0, 0.03, 1, 0.96])

    # show_values(g)
    bar_count = 0
    for bars in g.containers:
        bar_count += 1
        g.bar_label(bars, label_type='center', rotation=90, fmt='%.2f')
        # _x = p.get_x() + p.get_width() / 2
        # _y = (p.get_y() + p.get_height() + (p.get_height() * 0.01)) / 2
        # g.text(_x, _y, p, va='center')

    plt.legend(loc='upper right', ncol=4)

    # ax1 = g.axes[0][0]
    # nways = np.unique(my_df.nway.values)

    plt.minorticks_on()
    # ax1.grid(axis='y', which='minor', color='#999999', linewidth=.3, alpha=0.3)

    # my_df.score = my_df.score.astype(float)

    # print(my_df.groupby(['nway', 'run_name']).mean().score.round(2))
    plt.suptitle(f'Identification accuracy for subjects 1-8 (higher is better)')
    plt.show()


def visualise_s2n3(df, iv, study, n):
    # IV should be a list of your comparison (different rois or different vox res)
    plt.cla()
    # chained = lambda l: list(itertools.chain(*l))
    sns.set()
    sns.set_context("paper", font_scale=1.4)

    # target_exps = chained([
    #                           f'study{study}_{n}-way_{metric}_{levels}',
    #                       ] for levels in iv)
    if study == 2:
        my_df = df[(df.nway == n) & (df.vox_res.isin(iv)) & (df.study == study)
                   ]
        var = 'vox_res'
    else:
        my_df = df[(df.nway == n) & (df.ROI.isin(iv)) & (df.study == study)
                   ]
        var = 'ROI'

    # fn = lambda x: '_'.join(x.split('_')[1:])
    # my_df.exp_name = my_df.exp_name.apply(fn)
    my_df = my_df.explode('score')
    # my_df.exp_name.unique()

    # g = sns.catplot(x='nway', y='score', hue='run_name', col='subject', kind="bar", data=my_df, legend=False,
    #                 legend_out=True, sharey=False)
    g = sns.barplot(data=my_df, x=var, y='score', hue='metric')
    g.axhline(1/n, color='k', linestyle='--')
    # g.set_titles(col_template='Subject')

    # for i in g.containers:
    #     g.bar_label(i,)

    show_values(g)

    plt.gcf().set_size_inches(8, 4)
    plt.gcf().tight_layout(rect=[0, 0.03, 1, 0.96])

    # ax1 = g.axes[0][0]
    # nways = np.unique(my_df.nway.values)

    plt.minorticks_on()
    # ax1.grid(axis='y', which='minor', color='#999999', linewidth=.3, alpha=0.3)

    # my_df.score = my_df.score.astype(float)

    # print(my_df.groupby(['nway', 'run_name']).mean().score.round(2))
    plt.suptitle(f'Identification accuracy (subjects pooled) in {n}-way (higher is better)')
    plt.show()


def visualise_s2(df, iv, study, n):
    # IV should be a list of your comparison (different rois or different vox res)
    plt.cla()

    my_df = df[(df.nway == n)]
    means = [my_df['comp_1_mean'],my_df['comp_2_mean']]
    var = 'vox_res'
    if study == 2:
        my_df = df[(df.nway == n) & (df.vox_res.isin(iv)) & (df.study == study)
                   ]
        var = 'vox_res'
    else:
        my_df = df[(df.nway == n) & (df.ROI.isin(iv)) & (df.study == study)
                   ]
        var = 'ROI'

    my_df = my_df.explode('score')

    g = sns.barplot(data=my_df, x=var, y='score', hue='metric')
    g.axhline(1/n, color='k', linestyle='--')
    # g.set_titles(col_template='Subject')

    # for i in g.containers:
    #     g.bar_label(i,)

    show_values(g)

    plt.gcf().set_size_inches(8, 4)
    plt.gcf().tight_layout(rect=[0, 0.03, 1, 0.96])

    # ax1 = g.axes[0][0]
    # nways = np.unique(my_df.nway.values)

    plt.minorticks_on()
    # ax1.grid(axis='y', which='minor', color='#999999', linewidth=.3, alpha=0.3)

    # my_df.score = my_df.score.astype(float)

    # print(my_df.groupby(['nway', 'run_name']).mean().score.round(2))
    plt.suptitle(f'Identification accuracy (subjects pooled) in {n}-way (higher is better)')
    plt.show()


def perm_test(perm_scores, observed_scores):
    mean_perm = np.mean(perm_scores)
    sd_perm = np.std(perm_scores)
    print('Permutation Mean: {} | STD: {}'.format(mean_perm, sd_perm))

    mean_observed = np.mean(observed_scores)

    p_obs = norm.pdf(mean_observed, loc=mean_perm, scale=sd_perm)

    critical = 0.05
    # correcting for 48 comparisons (most conservative correction)
    correction = critical / (6*8)

    sig = p_obs < correction

    return p_obs, sig


def grab_comp(df, study, nway=2, metric='LPIPS', subject=None, roi='VC', vox='1pt8mm'):
    # Gets dataframe of selected things
    if subject is None:
        subject = [1, 2, 3, 4, 5, 6, 7, 8]
    comp =(df[
        (df['study'] == study) & (df['nway'] == nway) & (df['metric'] == metric) & (df['subject'].isin(subject))
        & (df['ROI'] == roi) & (df['vox_res'] == vox)
               ])

    return comp


def grab_scores(df, study, nway=2, metric='LPIPS', roi='VC', vox='1pt8mm'):
    # Grabs only the scores of selected things.
    comp =(df[
        (df['study'] == study) & (df['nway'] == nway) & (df['metric'] == metric)
        & (df['ROI'] == roi) & (df['vox_res'] == vox)
               ])

    comp = comp.reset_index() # TODO: CHeck this doesn't break study 2

    scores = np.array(comp['score'][0])

    return scores


def group_by(df, run_name):
    explode = df.explode('score')
    run_names = explode['run_name']
    run_names_imp = run_names.agg({'run_name': lambda x: x.tolist()})
    implode = explode.groupby(['study'], as_index=False).agg({'run_name': lambda x: run_name,
                                                            'score': lambda x: x.tolist(),
                                                            'subject': lambda x: list(set(x.tolist())),
                                                            'study': 'first',
                                                            'vox_res': 'first',
                                                            'ROI': 'first',
                                                            'set_size': 'first',
                                                            'metric': 'first',
                                                            'nway': 'first',
                                                            'repeats': 'first',
                                                            })

    # CHECKS SCRIPT IS WORKING
    # print(len(implode['score'][0]))
    # explode_implode = implode.explode('score')
    # score_test = explode_implode['score'].to_numpy()
    # score_good = explode['score'].to_numpy()
    # np.array_equal(score_good, score_test)

    # s.map(lambda x: x.tolist()).to_numpy()
    return implode


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