import pandas as pd
import os
import matplotlib.pyplot as plt

def plot(setting="C", first_50=False, dir_path=None, forest=False):
    base_dir_path = 'results/experiments_benchmarking/ihdp'
    if setting == "C":
        if dir_path is not None:
            full_dir_path = os.path.join(base_dir_path, dir_path)
        else:
            full_dir_path = base_dir_path
        # Read the CSV file for C setting
        df = pd.read_csv(os.path.join(full_dir_path, 'results_C.csv'))
        if forest:
            Forest_df = pd.read_csv(os.path.join(full_dir_path, 'grf_original.csv'))
            Forest_df = Forest_df[Forest_df['simu'] <= 10]
            df['cf_out'] = Forest_df['cf_out']
            df['t_out'] = Forest_df['t_out']
    elif setting == "D":
        if dir_path is not None:
            full_dir_path = os.path.join(base_dir_path, dir_path)
        else:
            full_dir_path = base_dir_path
        # Read the CSV file for D setting
        df = pd.read_csv(os.path.join(full_dir_path, 'results_D.csv'))
        if forest:
            # Forest_df = pd.read_csv(os.path.join(full_dir_path, 'grf_modified.csv'))
            # Forest_df = Forest_df[Forest_df['simu'] <= 10]
            df['cf_out'] = Forest_df['cf_out']
            df['t_out'] = Forest_df['t_out']

    if first_50:
        # Sort by 'cate_var_out' and take top 50 rows
        df = df.sort_values(by='cate_var_out', ascending=True).head(50)

    # Get the lists
    cate_var_out = df['cate_var_out'].tolist()
    # TNet_out = df['TNet_out'].tolist()
    TARNet_out = df['TARNet_out'].tolist()
    # CF_out = df['cf_out'].tolist()
    # TRF_out = df['t_out'].tolist()
    TARNet_single_out = df['TARNet_single_out'].tolist()
    # TARNet_single_2_out = df['TARNet_single_2_out'].tolist()
    MLP_CATENet_out = df['MLP_CATENet_out'].tolist()

    plt.figure(figsize=(10, 8))

    # scatter plots
    # plt.scatter(cate_var_out, TNet_out, color='red', label='TNet')
    plt.scatter(cate_var_out, TARNet_out, color='green', label='TARNet')
    plt.scatter(cate_var_out, TARNet_single_out, color='purple', label='TARNet_single')
    # plt.scatter(cate_var_out, TARNet_single_2_out, color='black', label='TARNet_single_2')
    plt.scatter(cate_var_out, MLP_CATENet_out, color='black', label='MLP_CATENet_out')
    # plt.scatter(cate_var_out, CF_out, color='orange', label='CF')
    # plt.scatter(cate_var_out, TRF_out, color='blue', label='TRF')

    # labels
    plt.xlabel('Variance')
    plt.ylabel('RMSE')
    plt.legend()

    if not os.path.exists(f"plots/{dir_path}/"):
        os.mkdir(f"plots/{dir_path}/")

    if first_50:
        plt.savefig(f"plots/{dir_path}/IHDP_{setting}_first_50.png")
    else:
        plt.savefig(f"plots/{dir_path}/IHDP_{setting}.png")

# plot(setting="C", first_50=False, dir_path='comparison_tarnet_single_2', forest=False)
# plot(setting="C", first_50=True, dir_path='comparison_tarnet_single_2', forest=False)
plot(setting="C", first_50=False, dir_path='comparison', forest=False)
plot(setting="C", first_50=True, dir_path='comparison', forest=False)
# plot(setting="D", first_50=False)
# plot(setting="D", first_50=True)