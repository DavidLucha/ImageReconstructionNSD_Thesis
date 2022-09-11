# This script goes through all network folders with the tables and computes accuracy
# Either with n-way or pairwise and then spits out a master table for each metric, showing accuracies
# Over each repeat or for each reconstruction (pairwise)

import pandas as pd
import os
import time

from utils_2 import nway_comp, pairwise_comp

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
        network = folder.split(sep, 1)[0]
        print('Evaluating network: {}'.format(network))

        # Load data
        print('Reading Data')
        # TODO: Do I nead header here? No.
        pcc = pd.read_csv(os.path.join(folder_dir, 'pcc_table.csv'), index_col=0)
        # ssim = pd.read_excel(os.path.join(folder_dir, 'ssim_table.xlsx'), engine='openpyxl', index_col=0)
        lpips = pd.read_csv(os.path.join(folder_dir, 'lpips_table.csv'), index_col=0)
        # I know this is redundant but...
        pcc = pcc.to_numpy()
        lpips = lpips.to_numpy()
        print('Data Ready')

        if count == 1:
            print('Saving reconstruction names')
            recon_names = list(pcc.columns)

        if nway:
            print('Running n-way comparisons')
            for n in ns:
                way_label = '{}-way'.format(n)

                print('Running n-way comparisons')
                pcc_nway_out = nway_comp(pcc, n=n, repeats=repeats, metric="pcc")
                print('PCC Complete')
                # ssim_nway_out = nway_comp(ssim, n=n, repeats=repeats, metric="ssim")
                # print('SSIM Complete')
                lpips_nway_out = nway_comp(lpips, n=n, repeats=repeats, metric="lpips")
                print('LPIPS Complete')

                # So, for n, add a dictionary that has the n as key, and another dictionary as the value
                # The second dictionary has run name as key and the accuracies of the repeats for values
                pcc_master[way_label][network] = pcc_nway_out
                # ssim_master[way_label][network] = ssim_nway_out
                lpips_master[way_label][network] = lpips_nway_out

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

        # if count == 8:
        #     # raise Exception('check masters')
        #     break

    # Setup writers
    # TODO: Change this to appropriate file name
    # TODO: Test saving CSV
    # pcc_writer = pd.ExcelWriter(os.path.join(save_path, "pcc_master_pairwise_out.csv"))
    # ssim_writer = pd.ExcelWriter(os.path.join(save_path, "ssim_master_pairwise_out.xlsx"))
    # lpips_writer = pd.ExcelWriter(os.path.join(save_path, "lpips_master_pairwise_out.csv"))

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


if __name__ == "__main__":
    main()