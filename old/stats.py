from utils_2 import eval_grab, load_masters
import eval_utils as nets

import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np





# ------------ SAVE DATA IN SPSS FRIENDLY FORMAT ---------------- #

# Loads file
# give directory, metric, comparison_type, list of column (network name)
# if list, combine
master_root = 'D:/Lucha_Data/final_networks/output/'

# Load master sheets here, otherwise have to reload
# nway_master = load_masters(master_root, "nway")
pairwise_master = load_masters(master_root, "pairwise")

master = {'trial': [],
          'study': [],
          'subj': [],
          'vox_res': [],
          'ROI': [],
          'set_size': [],
          'recon': [],
          'PCC': [],
          'LPIPS': []
          }

df_master = pd.DataFrame(master)

full_list = []

for (columnNamePCC, columnDataPCC), (_, columnDataLPIPS) in zip(pairwise_master['pcc'].iteritems(), pairwise_master['lpips'].iteritems()):
    # run_name = str(columnName).split('_')
    vals = {'PCC': columnDataPCC, 'LPIPS': columnDataLPIPS}
    column_df = pd.DataFrame(data=vals).reset_index()

    study, subj, vox_res, ROI, set_size = str(columnNamePCC).split('_')

    column_df['trial'] = column_df.index + 1
    column_df['study'] = study
    column_df['subj'] = subj
    column_df['vox_res'] = vox_res
    column_df['ROI'] = ROI
    column_df['set_size'] = set_size
    column_df['recon'] = columnDataPCC.index.values.tolist()

    # Rearrange
    column_df = column_df[['trial','study', 'subj', 'vox_res', 'ROI', 'set_size', 'recon', 'PCC', 'LPIPS']]

    full_list.append(column_df)

df_master = pd.concat(full_list)

save_path = 'D:/Lucha_Data/final_networks/output/'
df_master.to_csv(os.path.join(save_path, "Full_Data_SPSS.csv"))

# ------------ END ---------------- #

    # row = {# 'recon': recon,
    #        'study': study,
    #        'subj': subj,
    #        'vox_res': vox_res,
    #        'ROI': ROI,
    #        'set_size': set_size
    #        }
        # df_master = df_master.append(row, ignore_index=True)
    # recon = columnData.index[]

# TESTING 7500 vs 1200 at 1.8mm.
comp_1 = eval_grab(pairwise_master, nets.data_1pt8mm_7500)
comp_2 = eval_grab(pairwise_master, nets.data_1pt8mm_1200)



pcc_diffs = comp_1['pcc'] - comp_2['pcc']
lpips_diffs = comp_1['lpips'] - comp_2['lpips']

kwargs = dict(alpha=0.2, bins=50)

plt.hist(comp_1['lpips'], **kwargs, color='b', label='LPIPS')

plt.hist(pcc_diffs, **kwargs, color='g', label='PCC')
plt.hist(lpips_diffs, **kwargs, color='b', label='LPIPS')
plt.axvline(pcc_diffs.mean(), color='r', linestyle='dashed', linewidth=1)
plt.axvline(lpips_diffs.mean(), color='y', linestyle='dashed', linewidth=1)
plt.gca().set(title='Distribution of Accuracy Differences', ylabel='Frequency')
plt.legend()
plt.show()

plt.cla()
plt.close('all')

# Convert differences to z scores
z_lpips = stats.zscore(lpips_diffs)

# Boolean for those with z scores exceeding 3
outlier = abs(z_lpips >= 3)

# Calculate cutoff rawscore based on z score of 3
cutoff = (lpips_diffs.mean() + (3 * lpips_diffs.std()))

print('Count of outliers = ', np.count_nonzero(outlier))

# Remove outliers for raw diffs
lpips_diffs_cleaned = lpips_diffs[lpips_diffs < cutoff]

print('Original differences: \n'
      '----- Mean: {}\n'
      '----- STD: {}\n'
      'Cleaned differences: \n'
      '----- Mean: {}\n'
      '----- STD: {}'.format(lpips_diffs.mean(), lpips_diffs.std(), lpips_diffs_cleaned.mean(),
                             lpips_diffs_cleaned.std()))

stats.wilcoxon(comp_1['lpips'], comp_2['lpips'])

# Save all stats columns

# Loads file
# give directory, metric, comparison_type, list of column (network name)
# if list, combine
master_root = 'D:/Lucha_Data/final_networks/output/'

# Load master sheets here, otherwise have to reload
# nway_master = load_masters(master_root, "nway")
pairwise_master = load_masters(master_root, "pairwise")

test_list = [
    nets.data_subj_01,
    nets.data_subj_02,
    nets.data_subj_03,
    nets.data_subj_04,
    nets.data_subj_05,
    nets.data_subj_06,
    nets.data_subj_07,
    nets.data_subj_08,
    nets.data_1pt8mm_1200,
    nets.data_1pt8mm_4000,
    nets.data_1pt8mm_7500,
    nets.data_3mm_1200,
    nets.data_3mm_4000,
    nets.data_3mm_7500,
    nets.data_V1toV3,
    nets.data_V1toV3nHVC,
    nets.data_V1toV3nRAND,
    nets.data_HVC
]

pcc_full = {}
lpips_full = {}
header = []

for test in test_list:
    header.append(str(test).replace('nets.', ''))
    comp = eval_grab(pairwise_master, test)
    pcc_full[str(test)] = (comp['pcc'])
    lpips_full[str(test)] = (comp['lpips'])

pcc_df = pd.concat(pcc_full)
table_lpips_pd.to_csv(os.path.join(save_path, "lpips_table.csv"))

exit(200000)

# subj_01_stacked = subj_01.unstack().reset_index(drop=True)
# subj_01_stack2 = subj_01.iloc[:,0]

nets.data_subj_01
nets.data_subj_02
nets.data_subj_03
nets.data_subj_04
nets.data_subj_05
nets.data_subj_06
nets.data_subj_07
nets.data_subj_08
# Study 2
# 8 Participants combined
# Not really a direct comp here, just need the combined for each six conditions
# 1200, 4000, 7500 @ 1.8- and 3mm
nets.data_1pt8mm_1200
nets.data_1pt8mm_4000
nets.data_1pt8mm_7500

nets.data_3mm_1200
nets.data_3mm_4000
nets.data_3mm_7500

# Study 3
# 8 participants combined
# V1-V3 vs HVC
# V1-V3 vs V1-V3 + HVC
# V1-V3 + HVC vs V1-V3 + Rand
# V1-V3 vs V1-V3 + Rand

nets.data_V1toV3
nets.data_V1toV3nHVC
nets.data_V1toV3nRAND
nets.data_HVC
