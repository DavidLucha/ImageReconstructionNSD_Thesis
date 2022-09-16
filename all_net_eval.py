# This script goes through all network folders with the tables and computes accuracy
# Either with n-way or pairwise and then spits out a master table for each metric, showing accuracies
# Over each repeat or for each reconstruction (pairwise)
import numpy as np
import pandas as pd
import os
import time

import matplotlib.pyplot as plt

from utils_2 import nway_comp, pairwise_comp, permutation
from eval_utils import permute, perm_test, grab_comp, group_by, visualise_s1, visualise_s2n3, bootstrap_acc, \
    save_20images

# Walk through folders in all eval network folders
# Get scores from each trained network into dataframe
def main():
    # TODO: Change this for laptop
    root_dir = 'D:/Lucha_Data/final_networks/output/'
    # root_dir = 'C:/Users/david/Documents/Thesis/final_networks/output/'  # FOR laptop
    networks = os.path.join(root_dir,'all_eval/')
    save_path = root_dir
    recon_out_path = os.path.join(root_dir,'recons_out/')

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

        # Save the selected images out
        # real_path = os.path.join(folder_dir, 'images/real/')
        # recon_path = os.path.join(folder_dir, 'images/recon/')

        # save_20images(in_path=real_path, out_path=recon_out_path, run_name=folder, type='real')
        # save_20images(in_path=recon_path, out_path=recon_out_path, run_name=folder, type='recon')

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

        # TODO: Remove
        # if count == 1:
        #     raise Exception('check masters')
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


def save_recons():
    root_dir = 'D:/Lucha_Data/final_networks/output/'
    recon_out_path = 'D:/Lucha_Data/final_networks/output/recons_out/'
    networks = os.path.join(root_dir, 'all_eval/')
    count = 0

    for folder in os.listdir(networks):
        start = time.time()
        count += 1

        # print(folder)
        folder_dir = os.path.join(networks, folder)

        real_path = os.path.join(folder_dir, 'images/real/')
        recon_path = os.path.join(folder_dir, 'images/recon/')

        save_20images(in_path=real_path, out_path=recon_out_path, run_name=folder, type='real')
        save_20images(in_path=recon_path, out_path=recon_out_path, run_name=folder, type='recon')

        # TODO: REMOVE THIS
        if count == 2:
            break


if __name__ == "__main__":
    # Main spits out the main dataframe with the scores
    # TODO: run main first. DONE.
    # df = main()

    # exit(69)

    save_dir = 'D:/Lucha_Data/final_networks/output/' # TODO: Change this.
    # df.to_pickle(os.path.join(save_dir, 'test_perm.pkl'))

    full_df = pd.read_pickle('D:/Lucha_Data/final_networks/output/full_dataset.pkl')

    # ------------- ADD STUDY 1 P VALUES ------------- #

    def study_1():
        # Even with 500 with are getting stupid significant. Run 100k only if needed.
        df = permute(1000)  # TODO: RUN with 100k

        copy = full_df

        # add permutation scores to the main df
        copy = copy.assign(perm_score=df['score'])

        # set up p dictionary
        values = dict(p_values=[], sig=[], mean_acc=[], std_acc=[], CI_LB=[], CI_UB=[], CI_diff_low=[], CI_diff_high=[])

        # Go through all study 1 experiments and get p value.
        for id, row in copy[copy['study']==1].iterrows():
            p, sig = perm_test(row['perm_score'], row['score'])
            values['p_values'].append(p)
            values['sig'].append(sig)

            # Calculate bootstrapped means
            observations = len(row['score'])
            # print(observations)
            CI, CI_diffs = bootstrap_acc(row['score'], permutations=10000, observations=observations) # TODO: 10k perms
            values['CI_LB'].append(CI[0])
            values['CI_UB'].append(CI[1])
            values['CI_diff_low'].append(CI_diffs[0])
            values['CI_diff_high'].append(CI_diffs[1])

            mean_acc = np.mean(row['score'])
            std_acc = np.std(row['score'])
            values['mean_acc'].append(mean_acc)
            values['std_acc'].append(std_acc)

        values_df = pd.DataFrame.from_dict(values)

        copy = copy.assign(p_value=values_df.p_values, significance=values_df.sig, mean_acc=values_df.mean_acc,
                           std_acc=values_df.std_acc, CI_LB=values_df.CI_LB, CI_UB=values_df.CI_UB,
                           CI_diff_low=values_df.CI_diff_low, CI_diff_high=values_df.CI_diff_high)

        copy = copy[copy['study']==1]

        copy.to_pickle(os.path.join(save_dir, 'full_data_study_1.pkl'))

        return copy

    # study1_out = study_1()

    # exit(69)


    # ------------- END ------------- #

    # Study 2 and 3

    full_df = pd.read_pickle('D:/Lucha_Data/final_networks/output/full_dataset.pkl')


    # --------------------------------- Data for Study 2/3 --------------------------------- #

    def study_2n3():
        # GRAB the full dataset (each run) and combine per condition of study 2 and 3
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
                # TODO: Remove
                # metric = 'LPIPS'
                # nway = 5
                run_count += 1
                # nway_label = '{}-way'.format(nway)
                comp = grab_comp(full_df, study=1, nway=nway, metric=metric, subject=all_subs, roi=ROI, vox=vox_res)
                # run_name last variable is IV
                run_name = 'study{}_{}-way_{}_{}'.format(study, nway, metric, vox_res)
                implode_comp = group_by(comp, run_name)

                mean_acc = np.mean(implode_comp['score'][0])
                std_acc = np.std(implode_comp['score'][0])
                implode_comp['mean_acc'] = mean_acc

                observations = len(implode_comp['score'][0])

                CI, CI_diffs = bootstrap_acc(implode_comp['score'][0], permutations=10000, observations=observations)  # TODO: Check this
                implode_comp['CI_LB'] = CI[0]
                implode_comp['CI_UP'] = CI[1]
                implode_comp['CI_diff_low'] = CI_diffs[0]
                implode_comp['CI_diff_high'] = CI_diffs[1]

                # print(len(implode_comp['score'][0]))
                # test = implode_comp['score']

                implode_comp['std_acc'] = std_acc

                # implode_comp['CI'] = 1.04
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
                # nway_label = '{}-way'.format(nway)
                comp = grab_comp(full_df, study=2, nway=nway, metric=metric, subject=all_subs, roi=ROI, vox=vox_res)
                # run_name last variable is IV
                run_name = 'study{}_{}-way_{}_{}'.format(study, nway, metric, vox_res)
                implode_comp = group_by(comp, run_name)

                mean_acc = np.mean(implode_comp['score'][0])
                std_acc = np.std(implode_comp['score'][0])
                implode_comp['mean_acc'] = mean_acc

                observations = len(implode_comp['score'][0])

                CI, CI_diffs = bootstrap_acc(implode_comp['score'][0], permutations=10000,
                                             observations=observations)  # TODO: Check this
                implode_comp['CI_LB'] = CI[0]
                implode_comp['CI_UP'] = CI[1]
                implode_comp['CI_diff_low'] = CI_diffs[0]
                implode_comp['CI_diff_high'] = CI_diffs[1]

                # print(len(implode_comp['score'][0]))
                # test = implode_comp['score']

                implode_comp['std_acc'] = std_acc

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
                    # nway_label = '{}-way'.format(nway)
                    comp = grab_comp(full_df, study=study, nway=nway, metric=metric, subject=all_subs, roi=area, vox=vox_res)
                    # run_name last variable is IV
                    run_name = 'study{}_{}-way_{}_{}'.format(study, nway, metric, area)
                    implode_comp = group_by(comp, run_name)

                    mean_acc = np.mean(implode_comp['score'][0])
                    std_acc = np.std(implode_comp['score'][0])
                    implode_comp['mean_acc'] = mean_acc

                    observations = len(implode_comp['score'][0])

                    CI, CI_diffs = bootstrap_acc(implode_comp['score'][0], permutations=10000,
                                                 observations=observations)  # TODO: Check this
                    implode_comp['CI_LB'] = CI[0]
                    implode_comp['CI_UP'] = CI[1]
                    implode_comp['CI_diff_low'] = CI_diffs[0]
                    implode_comp['CI_diff_high'] = CI_diffs[1]

                    # print(len(implode_comp['score'][0]))
                    # test = implode_comp['score']

                    implode_comp['std_acc'] = std_acc

                    if run_count == 1:
                        data_df = implode_comp
                    else:
                        data_df = pd.concat([data_df, implode_comp])

        data_df = data_df.reset_index(drop=True)

        data_df.to_pickle(os.path.join(save_dir, 'full_data_study_2n3.pkl'))

        return data_df

    study2_data = study_2n3()

    # --------------------------------- END data for Study 2/3 --------------------------------- #
    # raise Exception('check')
    exit(0)

    # ------------ CALCULATE AND ADD CIs ------------ #

    # --------------------------------- CI Calculation --------------------------------- #

    study_1_data = pd.read_pickle('D:/Lucha_Data/final_networks/output/full_data_study_1.pkl')

    # study_1_explode = study_1_data.explode('score')
    # study_1_vis = study_1_explode.groupby('metric')
    visualise_s1(study_1_data)

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

