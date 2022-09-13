# This script goes through all network folders with the tables and computes accuracy
# Either with n-way or pairwise and then spits out a master table for each metric, showing accuracies
# Over each repeat or for each reconstruction (pairwise)

import pandas as pd
import os
import time
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import itertools

from scipy.stats import norm

from utils_2 import nway_comp, pairwise_comp, permutation

# Walk through folders in all eval network folders
def main():
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
    repeats = 1000

    # Empty list to house all networks pcc evals
    pcc_master = {}
    # ssim_master = {}
    lpips_master = {}

    # Set up dict of dicts (pairwise)
    pcc_master['pairwise'] = {}
    # ssim_master['pairwise'] = {}
    lpips_master['pairwise'] = {}

    # Add new evaluation requests
    data = {
        **{'run_name': [], 'score': [], 'study': [], 'subject': [], 'vox_res': [], 'ROI': [], 'set_size': [],
           'metric': [], 'nway': [], 'repeats': []}
    }

    # Set up dict of dicts (nway)
    for n in ns:
        way_label = '{}-way'.format(n)
        pcc_master[way_label] = {}
        # ssim_master[way_label] = {}
        lpips_master[way_label] = {}

    for folder in os.listdir(networks):
        start = time.time()
        count += 1

        # print(folder)
        folder_dir = os.path.join(networks, folder)

        # Get name of network
        sep = '_Stage3_'
        network = folder.split(sep, 1)[0]  # aka run_name

        if 'V1_to_V3_n_rand' in network:
            network = network.replace('V1_to_V3_n_rand','V1toV3nRand')
        elif 'V1_to_V3_n_HVC' in network:
            network = network.replace('V1_to_V3_n_HVC','V1toV3nHVC')
        elif 'V1_to_V3_max' in network:
            network = network.replace('V1_to_V3', 'V1toV3')

        study, subj, vox_res, ROI, set_size = str(network).split('_')
        study = int(study.split('y', 1)[1])
        subj = int(subj.split('0', 1)[1])
        eval_options = dict(study=study, subject=subj, vox_res=vox_res, ROI=ROI, set_size=set_size, repeats=repeats)

        print('Evaluating network: {}'.format(network))

        # Load data
        print('Reading Data')
        # TODO: Do I nead header here? No.
        pcc = pd.read_csv(os.path.join(folder_dir, 'pcc_table.csv'), index_col=0)
        # ssim = pd.read_excel(os.path.join(folder_dir, 'ssim_table.xlsx'), engine='openpyxl', index_col=0)
        lpips = pd.read_csv(os.path.join(folder_dir, 'lpips_table.csv'), index_col=0)

        if count == 1:
            print('Saving reconstruction names')
            # Save the names if we want to include a copy of recons.
            recon_names = list(pcc.columns)

        # I know this is redundant but...
        pcc = pcc.to_numpy()
        lpips = lpips.to_numpy()
        print('Data Ready')

        if nway:
            print('Running n-way comparisons')
            for n in ns:
                way_label = '{}-way'.format(n)

                print('Running {} comparisons'.format(way_label))
                pcc_nway_out = nway_comp(pcc, n=n, repeats=repeats, metric="pcc")
                print('PCC Complete')

                # ssim_nway_out = nway_comp(ssim, n=n, repeats=repeats, metric="ssim")
                # print('SSIM Complete')
                lpips_nway_out = nway_comp(lpips, n=n, repeats=repeats, metric="lpips")
                print('LPIPS Complete')

                scores = [pcc_nway_out, lpips_nway_out]
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

                # So, for n, add a dictionary that has the n as key, and another dictionary as the value
                # The second dictionary has run name as key and the accuracies of the repeats for values
                # pcc_master[way_label][network] = pcc_nway_out
                # ssim_master[way_label][network] = ssim_nway_out
                # lpips_master[way_label][network] = lpips_nway_out

                print('Evaluations saved to master list.')

        if pairwise:
            # pcc_nway_out = nway_comp(data, n=2, repeats=10, metric="pcc")
            print('Running pairwise comparisons')
            pcc_pairwise_out = pairwise_comp(pcc, metric="pcc")
            print('PCC Complete')
            # ssim_pairwise_out = pairwise_comp(ssim, metric="ssim")
            # print('SSIM Complete')
            lpips_pairwise_out = pairwise_comp(lpips, metric="lpips")
            print('LPIPS Complete')

            pcc_master['pairwise'][network] = pcc_pairwise_out
            # ssim_master['pairwise'][network] = ssim_pairwise_out
            lpips_master['pairwise'][network] = lpips_pairwise_out
            print('Evaluations saved to master list.')

        end = time.time()
        print('Time per run =', end - start)
        # if count == 1:
        #     break

        # if count == 8:
        #     # raise Exception('check masters')
        #     break

    # Setup writers
    # TODO: Change this to appropriate file name
    # TODO: Test saving CSV
    # pcc_writer = pd.ExcelWriter(os.path.join(save_path, "pcc_master_pairwise_out.csv"))
    # ssim_writer = pd.ExcelWriter(os.path.join(save_path, "ssim_master_pairwise_out.xlsx"))
    # lpips_writer = pd.ExcelWriter(os.path.join(save_path, "lpips_master_pairwise_out.csv"))
    save = False

    if save:
        if pairwise:
            pcc_save = pd.DataFrame(pcc_master['pairwise'], index=recon_names)
            # ssim_save = pd.DataFrame(ssim_master['pairwise'], index=recon_names)
            lpips_save = pd.DataFrame(lpips_master['pairwise'], index=recon_names)
            print('Dataframes established.')

            print('Saving data...')
            # with pd.ExcelWriter(os.path.join(save_path, "pcc_master_out.xlsx")) as writer:
            pcc_save.to_csv(os.path.join(save_path, "pcc_master_pairwise_out.csv"))
            # with pd.ExcelWriter(os.path.join(save_path, "ssim_master_out.xlsx")) as writer:
            # ssim_save.to_excel(ssim_writer, sheet_name='pairwise_comparison')
            # with pd.ExcelWriter(os.path.join(save_path, "lpips_master_out.xlsx")) as writer:
            lpips_save.to_csv(os.path.join(save_path, "lpips_master_pairwise_out.csv"))

            # pcc_save.to_excel(os.path.join(save_path, "pcc_master_out.xlsx"))
            # ssim_save.to_excel(os.path.join(save_path, "ssim_master_out.xlsx"))
            # lpips_save.to_excel(os.path.join(save_path, "lpips_master_out.xlsx"))

        if nway:
            print('Saving data...')
            # with pd.ExcelWriter(os.path.join(save_path, "pcc_master_out.xlsx")) as writer:
            for n in ns:
                # TODO: Test working with csv writing
                way_label = '{}-way'.format(n)

                pcc_list = pd.DataFrame(pcc_master[way_label])
                pcc_list.to_csv(os.path.join(save_path, 'pcc_master_{}_comparison_out.csv'.format(way_label)))

                # ssim_list = pd.DataFrame(ssim_master[way_label])
                # ssim_list.to_excel(ssim_writer, sheet_name='{}_comparison'.format(way_label))

                lpips_list = pd.DataFrame(lpips_master[way_label])
                lpips_list.to_csv(os.path.join(save_path, 'lpips_master_{}_comparison_out.csv'.format(way_label)))

    # pcc_writer.save()
    # ssim_writer.save()
    # lpips_writer.save()

    # pcc_writer.close()
    # ssim_writer.close()
    # lpips_writer.close()
    print('Complete.')
    df = pd.DataFrame.from_dict(data)
    df.to_pickle(os.path.join(save_path, 'full_dataset.pkl'))
    return df


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

    if subject is None:
        subject = [1, 2, 3, 4, 5, 6, 7, 8]
    comp =(df[
        (df['study'] == study) & (df['nway'] == nway) & (df['metric'] == metric) & (df['subject'].isin(subject))
        & (df['ROI'] == roi) & (df['vox_res'] == vox)
               ])

    return comp


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


if __name__ == "__main__":
    # Main spits out the main dataframe with the scores
    # df = main()

    save_dir = 'D:/Lucha_Data/final_networks/output/'
    # df.to_pickle(os.path.join(save_dir, 'test_perm.pkl'))

    full_df = pd.read_pickle('D:/Lucha_Data/final_networks/output/full_dataset.pkl')

    # ------------- ADD STUDY 1 P VALUES ------------- #

    def study_1():
        # Even with 500 with are getting stupid significant. Run 100k only if needed.
        df = permute(500)

        copy = full_df

        # add permutation scores to the main df
        copy = copy.assign(perm_score=df['score'])

        # set up p dictionary
        p_values = dict(p_values=[], sig=[])

        # Go through all study 1 experiments and get p value.
        for id, row in copy[copy['study']==1].iterrows():
            p, sig = perm_test(row['perm_score'], row['score'])
            p_values['p_values'].append(p)
            p_values['sig'].append(sig)

        p_values_df = pd.DataFrame.from_dict(p_values)

        copy = copy.assign(p_value=p_values_df.p_values, significance=p_values_df.sig)

        copy.to_pickle(os.path.join(save_dir, 'full_data_study_1.pkl'))

    # ------------- END ------------- #

    # Study 2 and 3

    full_df = pd.read_pickle('D:/Lucha_Data/final_networks/output/full_dataset.pkl')


    # --------------------------------- Data for Study 2/3 --------------------------------- #

    def study_2n3():
        # Hyperparameters
        all_subs = list(range(1, 9))
        metrics = ['PCC', 'LPIPS']
        nways = [2, 5, 10]
        ROI = 'VC'

        run_count = 0
        # ------------ Study 2 ------------ #
        # All subjects at 1.8mm and 3mm.
        # And/or each subject at 1.8mm and 3mm.
        # @ each nway, for each metric (6 comparisons)
        # Still study 2 but get the scores from study 1
        study = 1
        vox_res = '1pt8mm'

        for metric in metrics:
            for nway in nways:
                run_count += 1
                comp = grab_comp(full_df, study=1, nway=nway, metric=metric, subject=all_subs, roi=ROI, vox=vox_res)
                # run_name last variable is IV
                run_name = 'study{}_{}-way_{}_{}'.format(study, nway, metric, vox_res)
                implode_comp = group_by(comp, run_name)
                if run_count == 1:
                    data_df = implode_comp
                else:
                    data_df = pd.concat([data_df, implode_comp])

        # rename study 1 part
        # test = data_df
        data_df['run_name'] = data_df['run_name'].str.replace('study1','study2')
        data_df['study'] = data_df['study'].replace(1, 2)

        study = 2
        vox_res = '3mm'

        for metric in metrics:
            for nway in nways:
                run_count += 1
                comp = grab_comp(full_df, study=2, nway=nway, metric=metric, subject=all_subs, roi=ROI, vox=vox_res)
                # run_name last variable is IV
                run_name = 'study{}_{}-way_{}_{}'.format(study, nway, metric, vox_res)
                implode_comp = group_by(comp, run_name)
                if run_count == 1:
                    data_df = implode_comp
                else:
                    data_df = pd.concat([data_df, implode_comp])

        # data_df.astype({'study': 'int32', 'nway': 'int32', 'repeats': 'int32'}).dtypes


        # ------------ STUDY 3 ------------ #
        study = 3
        vox_res = '1pt8mm'
        ROI = ['V1toV3', 'HVC', 'V1toV3nHVC', 'V1toV3nRand']

        for area in ROI:
            for metric in metrics:
                for nway in nways:
                    run_count += 1
                    comp = grab_comp(full_df, study=study, nway=nway, metric=metric, subject=all_subs, roi=area, vox=vox_res)
                    # run_name last variable is IV
                    run_name = 'study{}_{}-way_{}_{}'.format(study, nway, metric, area)
                    implode_comp = group_by(comp, run_name)
                    if run_count == 1:
                        data_df = implode_comp
                    else:
                        data_df = pd.concat([data_df, implode_comp])

        data_df.to_pickle(os.path.join(save_dir, 'full_data_study_2n3.pkl'))

    # --------------------------------- END data for Study 2/3 --------------------------------- #

    visualise_s1(full_df)

    data_df = pd.read_pickle('D:/Lucha_Data/final_networks/output/full_data_study_2n3.pkl')
    study2_iv = ['1pt8mm', '3mm']
    visualise_s2n3(data_df, iv=study2_iv, study=2, n=5)
    study3_iv = ['V1toV3', 'HVC']
    visualise_s2n3(data_df, iv=study3_iv, study=3, n=5)
    study3_iv = ['V1toV3', 'V1toV3nHVC', 'V1toV3nRand']
    visualise_s2n3(data_df, iv=study3_iv, study=3, n=5)

    # TODO: new method for single subject study 2 and 3 comparisons
    # TODO: draw mean line
    # TODO: write save image code

