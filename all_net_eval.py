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


def visualise(df):
    chained = lambda l: list(itertools.chain(*l))
    sns.set()
    sns.set_context("paper", font_scale=1.4)

    target_exps = chained([
                              f'Study1_SUBJ0{sbj_num}_1pt8mm_VC_max',
                          ] for sbj_num in [1, 2, 3, 4, 5, 6, 7, 8])

    my_df = df[(df.nway > 0) & (df.nway <= 1000) & (df.repeats.isin([50])) &
               (df.run_name.isin(target_exps)) & (df.metric == 'LPIPS')
               ]

    # fn = lambda x: '_'.join(x.split('_')[1:])
    # my_df.exp_name = my_df.exp_name.apply(fn)
    my_df = my_df.explode('score')
    # my_df.exp_name.unique()

    # g = sns.catplot(x='nway', y='score', hue='run_name', col='subject', kind="bar", data=my_df, legend=False,
    #                 legend_out=True, sharey=False)
    g = sns.barplot(data=my_df, x="nway", y='score', hue='subject', n_boot=50)
    # g.set_titles(col_template='Subject')

    plt.gcf().set_size_inches(8, 4)
    plt.gcf().tight_layout(rect=[0, 0.03, 1, 0.96])

    # ax1 = g.axes[0][0]
    nways = np.unique(my_df.nway.values)

    plt.minorticks_on()
    # ax1.grid(axis='y', which='minor', color='#999999', linewidth=.3, alpha=0.3)

    my_df.score = my_df.score.astype(float)

    print(my_df.groupby(['nway', 'run_name']).mean().score.round(2))
    plt.suptitle(f'Identification accuracy for subjects 1-4 (higher is better)')
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

    # visualise(df)

    full_df = pd.read_pickle('D:/Lucha_Data/final_networks/output/full_dataset.pkl')

    comp_1 = full_df[
        (full_df['study'] == 1) & (full_df['nway'] == 5) & (full_df['metric'] == 'LPIPS')
               ]
    comp_2 = full_df[
        (full_df['study'] == 2) & (full_df['nway'] == 5) & (full_df['metric'] == 'LPIPS')
               ]

    comp_1_explode = comp_1.explode('score')
    comp_2_explode = comp_2.explode('score')

    score_1 = comp_1_explode['score'].to_numpy()
    score_2 = comp_2_explode['score'].to_numpy()

    diffs = score_1 - score_2

    # sub_1 = comp_1_explode[]

    # min(comp_1_explode.score)

    kwargs = dict(alpha=0.2, bins=50)

    plt.hist(comp_1_explode['score'], **kwargs, color='b', label='LPIPS')

    # diffs
    plt.hist(diffs, **kwargs, color='g', label='LPIPS')
    # plt.hist(lpips_diffs, **kwargs, color='b', label='LPIPS')
    plt.axvline(diffs.mean(), color='r', linestyle='dashed', linewidth=1)
    # plt.axvline(lpips_diffs.mean(), color='y', linestyle='dashed', linewidth=1)
    plt.gca().set(title='Distribution of Accuracy Differences', ylabel='Frequency')
    plt.legend()

    plt.show()

    plt.cla()
    plt.close('all')
    # comp_1_explode.reset_index()

    # permutation_list = dict(permutation_scores=[])
    # list=[12,3,4]
    # permutation_list['permutation_scores'].append(list)
    # perm_df = pd.DataFrame.from_dict(permutation_list)

    # print(dist)

