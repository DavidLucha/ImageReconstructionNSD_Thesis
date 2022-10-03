import matplotlib

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pingouin as pg

import numpy as np
import operator
import random
import os

matplotlib.style.use(matplotlib.get_data_path()+'/stylelib/apa.mplstyle')

from eval_utils import visualise_s2n3, visualise_s2, perm_test, grab_scores, non_param_paired_CI, bootstrap_acc


def study23_CIs():
    # Calculates the CIs for each of the study 2 and 3 comparisons
    df = pd.read_pickle('D:/Lucha_Data/final_networks/output/full_data_study_2n3.pkl')

    # df = df.reset_index()

    # set up p dictionary
    values = dict(mean_acc=[], std_acc=[], CI=[], CI_diffs=[])

    # Go through all study 1 experiments and get p value.
    for id, row in df.iterrows():
        # Calculate bootstrapped means
        CI, CI_diffs = bootstrap_acc(row['score'], permutations=100, observations=500)  # TODO: Check this
        values['CI'].append(CI)
        values['CI_diffs'].append(CI_diffs)

        mean_acc = np.mean(row['score'])
        std_acc = np.std(row['score'])
        values['mean_acc'].append(mean_acc)
        values['std_acc'].append(std_acc)

    values_df = pd.DataFrame.from_dict(values)

    df = df.assign(mean_acc=values_df.mean_acc, std_acc=values_df.std_acc, CI=values_df.CI, CI_diffs=values_df.CI_diffs)


def study2_analysis(df, dataframe, nways):
    base_p = 0.05

    # Study 2
    # Parameters
    study = 2
    # Testing voxel_resolution
    corrections = 6  # three nways, two metrics
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


            CI_diff = non_param_paired_CI(comp_1, comp_2, 0.95)
            # comp_1_mean_CI = bootstrap_acc(comp_1)
            # comp_2_mean_CI = bootstrap_acc(comp_2)
            # print(CI)

            p = wilcox.iloc[0]['p-val']
            t = wilcox.iloc[0]['W-val']
            sig = p < corrected_p

            dataframe['comparison'].append(comparison)
            dataframe['comp_1_label'].append(comp_labels[0])
            dataframe['comp_2_label'].append(comp_labels[1])
            dataframe['comp_1_scores'].append(comp_1)
            dataframe['comp_2_scores'].append(comp_2)
            dataframe['comp_diffs'].append(comp_1 - comp_2)
            dataframe['metric'].append(metric)
            dataframe['nway'].append(n)
            dataframe['comp_1_mean'].append(mean_1)
            dataframe['comp_1_mdn'].append(np.median(comp_1))
            dataframe['comp_1_std'].append(std_1)
            # dataframe['comp_1_mean_CI'].append(comp_1_mean_CI)
            dataframe['comp_2_mean'].append(mean_2)
            dataframe['comp_2_mdn'].append(np.median(comp_2))
            dataframe['comp_2_std'].append(std_2)
            # dataframe['comp_2_mean_CI'].append(comp_2_mean_CI)
            dataframe['mean_diff'].append(np.mean(comp_1 - comp_2))
            dataframe['mdn_diff'].append(np.median(comp_1 - comp_2))
            dataframe['CI_diff'].append(CI_diff)
            dataframe['t'].append(t)
            dataframe['p'].append(p)
            dataframe['RBC'].append(wilcox.iloc[0]['RBC'])
            dataframe['CLES'].append(wilcox.iloc[0]['CLES'])

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

    return dataframe, wilcox


def study3_wilcox(comp_1, comp_2, dataframe, comp_1_label, comp_2_label):
    mean_1 = np.mean(comp_1)
    std_1 = np.std(comp_1)

    mean_2 = np.mean(comp_2)
    std_2 = np.std(comp_2)

    wilcox = pg.wilcoxon(comp_1, comp_2, alternative='two-sided', correction='false')
    print(wilcox)

    CI_diff = non_param_paired_CI(comp_1, comp_2, 0.95)


    p = wilcox.iloc[0]['p-val']
    t = wilcox.iloc[0]['W-val']
    # sig = p < corrected_p

    dataframe['comp_1_label'].append(comp_1_label)
    dataframe['comp_2_label'].append(comp_2_label)
    dataframe['comp_1_scores'].append(comp_1)
    dataframe['comp_2_scores'].append(comp_2)
    dataframe['comp_diffs'].append(comp_1 - comp_2)
    dataframe['comp_1_mean'].append(mean_1)
    dataframe['comp_1_mdn'].append(np.median(comp_1))
    dataframe['comp_1_std'].append(std_1)
    # dataframe['comp_1_mean_CI'].append(comp_1_mean_CI)
    dataframe['comp_2_mean'].append(mean_2)
    dataframe['comp_2_mdn'].append(np.median(comp_2))
    dataframe['comp_2_std'].append(std_2)
    # dataframe['comp_2_mean_CI'].append(comp_2_mean_CI)
    dataframe['mean_diff'].append(np.mean(comp_1 - comp_2))
    dataframe['mdn_diff'].append(np.median(comp_1 - comp_2))
    dataframe['CI_diff'].append(CI_diff)
    dataframe['t'].append(t)
    dataframe['p'].append(p)
    dataframe['RBC'].append(wilcox.iloc[0]['RBC'])
    dataframe['CLES'].append(wilcox.iloc[0]['CLES'])

    return dataframe


def fried_append(dict, friedman, metric, n):
    dict['metric'].append(metric)
    dict['nway'].append(n)
    dict['W'].append(friedman.iloc[0]['W'])
    dict['df'].append(friedman.iloc[0]['ddof1'])
    dict['Q'].append(friedman.iloc[0]['Q'])
    dict['p'].append(friedman.iloc[0]['p-unc'])

    return dict


def study3_analysis(df, dataframe, nways):
    #
    df = orig_df[orig_df['study'] == 3]
    df = df.reset_index(drop=True)

    base_p = 0.05

    # Study 2
    # Parameters
    study = 3
    # Testing voxel_resolution
    corrections = 6
    corrected_p = base_p / corrections
    # effect = 'voxel resolution'
    metrics = ['LPIPS', 'PCC']
    comparison = 'ROI'
    comp_labels = ['V1toV3', 'HVC', 'V1toV3nHVC', 'V1toV3nRand']
    fried_list = ['V1toV3', 'V1toV3nHVC', 'V1toV3nRand']
    fried_data = dict(metric=[], nway=[], W=[], df=[], Q=[], p=[])

    for metric in metrics:
        for n in nways:
            # TODO: Remove these
            # metric = 'LPIPS'
            # n = 5
            # Do friedmans'
            sub_df = df[(df['metric'] == metric) & (df['nway'] == n) & (df['ROI'].isin(fried_list))]
            fried_df = sub_df[['run_name','score']]
            # Get names
            run_names = fried_df['run_name'].tolist()
            fried_df_t = fried_df.T[1:]
            fried_df_t.columns = run_names
            explode = fried_df_t.explode(['study3_{}-way_{}_V1toV3'.format(n, metric),
                                          'study3_{}-way_{}_V1toV3nHVC'.format(n, metric),
                                          'study3_{}-way_{}_V1toV3nRand'.format(n, metric)]).reset_index(drop=True)
            # explode.iloc[:,0]
            # TODO: Check precision of how we save these values
            explode = explode.astype(float)
            friedman_out = pg.friedman(explode)

            print(friedman_out)

            # add to dictionary
            fried_data = fried_append(fried_data, friedman_out, metric, n)

            # These zipped lists defined the comparison we want
            list_comp_1 = ['V1toV3', 'V1toV3nHVC', 'V1toV3nHVC', 'V1toV3nRand']
            list_comp_2 = ['HVC', 'V1toV3', 'V1toV3nRand', 'V1toV3']

            for big, little in zip(list_comp_1, list_comp_2):
                comp_1 = grab_scores(df, study=study, roi=big, metric=metric, nway=n)

                # 3mm
                comp_2 = grab_scores(df, study=study, roi=little, metric=metric, nway=n)

                dataframe = study3_wilcox(comp_1, comp_2, dataframe, big, little)
                dataframe['metric'].append(metric)
                dataframe['nway'].append(n)
                dataframe['comparison'].append(comparison)


    return dataframe, fried_data  # , wilcox


full_df = pd.read_pickle('D:/Lucha_Data/final_networks/output/full_dataset.pkl')
data_df = pd.read_pickle('D:/Lucha_Data/final_networks/output/full_data_study_2n3.pkl')

nways=[2,5,10]

save_path='D:/Lucha_Data/final_networks/output/'

# ------------------------ Study 2 ------------------------
study2 = False
if study2:
    data = dict(comparison=[], comp_1_label=[], comp_2_label=[], comp_1_scores=[], comp_2_scores=[], comp_diffs=[],
                metric=[], nway=[], comp_1_mean=[], comp_1_mdn=[], comp_1_std=[], comp_2_mean=[],
                comp_2_mdn=[], comp_2_std=[], mean_diff=[], mdn_diff=[], CI_diff=[], t=[], p=[], RBC=[], CLES=[])

    data_comp, wilcox = study2_analysis(data_df, data, nways)

    data_comp_df = pd.DataFrame.from_dict(data_comp)

    save_path = 'D:/Lucha_Data/final_networks/output/'
    data_comp_df.to_pickle(os.path.join(save_path, 'study2_wilcox_out.pkl'))

# print(matplotlib.get_data_path())


# ------------------------ STUDY 3 ------------------------
study3 = False

if study3:
    # Data here is only used for follow up, not for friedmans
    data_s3 = dict(comparison=[], comp_1_label=[], comp_2_label=[], comp_1_scores=[], comp_2_scores=[], comp_diffs=[],
                metric=[], nway=[], comp_1_mean=[], comp_1_mdn=[], comp_1_std=[], comp_2_mean=[],
                comp_2_mdn=[],comp_2_std=[], mean_diff=[], mdn_diff=[], CI_diff=[], t=[], p=[], RBC=[], CLES=[])

    orig_df = pd.read_pickle('D:/Lucha_Data/final_networks/output/full_data_study_2n3.pkl')
    study3_wilcox_out, study3_fried_out = study3_analysis(orig_df, data_s3, nways=nways)

    study3_wilcox_out_df = pd.DataFrame.from_dict(study3_wilcox_out)
    study3_fried_out_df = pd.DataFrame.from_dict(study3_fried_out)

    study3_wilcox_out_df.to_pickle(os.path.join(save_path, 'study3_wilcox_out.pkl'))
    study3_fried_out_df.to_pickle(os.path.join(save_path, 'study3_fried_out.pkl'))
    #TEST
    # df = pd.read_pickle('D:/Lucha_Data/final_networks/output/full_data_study_2n3.pkl')

# exit(69)


# ------------------------ SAVE ALL OUTPUTS AS CSVs ---------------------------
save_path='D:/Lucha_Data/final_networks/output/'

study1_out_df = pd.read_pickle(os.path.join(save_path, 'full_data_study_1.pkl'))
study2_wilcox_out_df = pd.read_pickle(os.path.join(save_path, 'study2_wilcox_out.pkl'))
study2n3_means_CIs_df = pd.read_pickle(os.path.join(save_path, 'full_data_study_2n3.pkl'))
study3_fried_out_df = pd.read_pickle(os.path.join(save_path, 'study3_fried_out.pkl'))
study3_wilcox_out_df = pd.read_pickle(os.path.join(save_path, 'study3_wilcox_out.pkl'))

# Add 'Subject' 'n-way' prefix to all values for graphing
study1_out_df['subject'] = 'Subject ' + study1_out_df['subject'].astype(str)
study1_out_df['nway'] = study1_out_df['nway'].astype(str) + '-way'

study2n3_means_CIs_df['nway'] = study2n3_means_CIs_df['nway'].astype(str) + '-way'
# study2n3_means_CIs_df['nway'] = study2n3_means_CIs_df['nway'].astype(str) + '-way'

# ------------------------ Assumption Checks ------------------------

assumption_checks = True

if assumption_checks:
    from scipy.stats import skew, kurtosis, shapiro, kstest
    import scipy.stats as stats
    import statsmodels.api as sm
    import statsmodels.stats as sm_stats
    # Get the shapiro-wilks p-value for each raw scores
    test_data = study2n3_means_CIs_df.iloc[0]['score']
    print(skew(test_data, bias=True))
    print(kurtosis(test_data, bias=True))
    print(shapiro(test_data))
    print(kstest(test_data, stats.norm.cdf))
    print(sm_stats.diagnostic.lilliefors(test_data))
    test_np = np.asarray(test_data)
    fig = sm.qqplot(test_np, line='45')
    plt.show()

    test_df = pd.DataFrame(test_data)

    file_out = os.path.join(save_path, 'normal_test_data.csv')

    test_df.to_csv(file_out, index=False)

    # Get all values for raw accuracy scores
    for id, row in study2n3_means_CIs_df.iterrows():
        data = row['score']
        print(row['run_name'])
        print('skew test result: ', skew(data, bias=True))
        print('kurtosis test result: ', kurtosis(data, bias=True))
        print('shapiro-wilks test result: ', shapiro(data))
        print('kolmogorov-smirnov with lilliefors result:', sm_stats.diagnostic.lilliefors(data))

    skew_list = []
    # Get the skewness for differences scores study 2
    for id, row in study2_wilcox_out_df.iterrows():
        plt.cla()
        data = row['comp_diffs']
        print('Stats for voxel resolution diff scores on {}-way using {}'.format(row['nway'], row['metric']))
        print('skew test result: ', skew(data, bias=True))
        skew_list.append(skew(data, bias=True))
        g = plt.hist(data, color='b', label='diff dist', bins=50)
        plt.show()

    # Get the skewness for differences scores study 3
    for id, row in study3_wilcox_out_df.iterrows():
        plt.cla()
        data = row['comp_diffs']
        print('Stats for diff between {} and {} on {}-way using {}'.format(row['comp_1_label'], row['comp_2_label'], row['nway'], row['metric']))
        print('skew test result: ', skew(data, bias=True))
        skew_list.append(skew(data, bias=True))
        g = plt.hist(data, color='b', label='diff dist', bins=50)
        plt.show()

    print(max(skew_list))
    print(min(skew_list))


with pd.ExcelWriter(os.path.join(save_path, 'all_studies_results.xlsx')) as writer:
    study1_out_df.to_excel(writer, sheet_name='Study 1')
    study2_wilcox_out_df.to_excel(writer, sheet_name='Study 2 Wilcox')
    study2n3_means_CIs_df.to_excel(writer, sheet_name='Study 2n3 CIs')
    study3_fried_out_df.to_excel(writer, sheet_name='Study 3 Friedmans')
    study3_wilcox_out_df.to_excel(writer, sheet_name='Study 3 Wilcox')

# print(np.mean(study1_out_df['score'][0]))


exit(69)


# Calculate bootstrapped means | perm
observations = len(study1_out_df['perm_score'][3])
_, _, bootstrap_perm = bootstrap_acc(study1_out_df['perm_score'][3], permutations=100, observations=observations) # TODO: 10k perms

observations = len(study1_out_df['score'][3])
_, _, bootstrap_real = bootstrap_acc(study1_out_df['score'][3], permutations=100, observations=observations) # TODO: 10k perms

kwargs = dict(alpha=0.2, bins=50)

plt.cla()

plt.hist(bootstrap_perm, **kwargs, color='b', label='LPIPS')
plt.hist(bootstrap_real, **kwargs, color='r', label='LPIPS')

plt.show()