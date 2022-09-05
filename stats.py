from utils_2 import eval_grab, load_masters
import eval_utils as nets

import pandas as pd
import os

# Loads file
# give directory, metric, comparison_type, list of column (network name)
# if list, combine
master_root = 'D:/Lucha_Data/final_networks/output/'

# Load master sheets here, otherwise have to reload
nway_master = load_masters(master_root, "nway")

# Study 1
# Each participant PCC, and LPIPS
# Testing against chance

test_comb = [
    "Study1_SUBJ01_1pt8mm_VC_max",
    "Study1_SUBJ02_1pt8mm_VC_max",
    "Study1_SUBJ03_1pt8mm_VC_max",
    "Study1_SUBJ04_1pt8mm_VC_max"
]

subj_01 = eval_grab(nway_master, test_comb)

subj_01_stacked = subj_01.unstack().reset_index(drop=True)
subj_01_stack2 = subj_01.iloc[:,0]

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





