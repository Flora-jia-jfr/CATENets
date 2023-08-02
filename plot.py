import pandas as pd
import os
import matplotlib.pyplot as plt

def IHDP_plot(setting="C", first_50=False, dir_path=None, forest=False):
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
    TARNet_single_2_out = df['TARNet_single_2_out'].tolist()
    # MLP_CATENet_out = df['MLP_CATENet_out'].tolist()

    plt.figure(figsize=(10, 8))

    # scatter plots
    # plt.scatter(cate_var_out, TNet_out, color='red', label='TNet')
    plt.scatter(cate_var_out, TARNet_out, color='green', label='TARNet')
    plt.scatter(cate_var_out, TARNet_single_out, color='purple', label='TARNet_single')
    plt.scatter(cate_var_out, TARNet_single_2_out, color='orange', label='TARNet_single_2')
    # plt.scatter(cate_var_out, MLP_CATENet_out, color='black', label='MLP_CATENet_out')
    # plt.scatter(cate_var_out, CF_out, color='orange', label='CF')
    # plt.scatter(cate_var_out, TRF_out, color='blue', label='TRF')

    # labels
    plt.xlabel('Variance')
    plt.ylabel('RMSE')
    plt.legend()

    if not os.path.exists(f"plots/{dir_path}/"):
        os.makedirs(f"plots/{dir_path}/")

    if first_50:
        plt.savefig(f"plots/{dir_path}/IHDP_{setting}_first_50.png")
    else:
        plt.savefig(f"plots/{dir_path}/IHDP_{setting}.png")

# IHDP_plot(setting="D", first_50=False, dir_path='comparison_tarnet_single_2', forest=False)
# IHDP_plot(setting="D", first_50=True, dir_path='comparison_tarnet_single_2', forest=False)
IHDP_plot(setting="C", first_50=False, dir_path='comparison', forest=False)
IHDP_plot(setting="C", first_50=True, dir_path='comparison', forest=False)
IHDP_plot(setting="D", first_50=False, dir_path='comparison', forest=False)
IHDP_plot(setting="D", first_50=True, dir_path='comparison', forest=False)
# IHDP_plot(setting="D", first_50=False)
# IHDP_plot(setting="D", first_50=True)

def ACIC2016_plot(data_dir='results/experiments_benchmarking/acic2016/comparsion', first_50=False):
    df = pd.read_csv(os.path.join(data_dir, 'results_False_2_4000.csv'))
    if first_50:
        df = df.sort_values(by='cate_var_out', ascending=True).head(50)
    cate_var_out = df['cate_var_out'].tolist()
    TARNet_out = df['TARNet_out'].tolist()
    TARNet_single_out = df['TARNet_single_out'].tolist()
    TARNet_single_2_out = df['TARNet_single_2_out'].tolist()

    plt.figure(figsize=(10, 8))

    # scatter plots
    plt.scatter(cate_var_out, TARNet_out, color='green', label='TARNet')
    plt.scatter(cate_var_out, TARNet_single_out, color='purple', label='TARNet_single')
    plt.scatter(cate_var_out, TARNet_single_2_out, color='orange', label='TARNet_single_2')

    # labels
    plt.xlabel('Variance')
    plt.ylabel('RMSE')
    plt.legend()

    if first_50:
        file_name = "ACIC_2016_first_50.png"
    else:
        file_name = "ACIC_2016.png"
    if not os.path.exists("plots/ACIC_2016/comparison/"):
        os.makedirs("plots/ACIC_2016/comparison/")

    plt.savefig(f"plots/ACIC_2016/comparison/{file_name}")

ACIC2016_plot()
ACIC2016_plot(first_50=True)

def twins_plot(data_dir='results/experiments_benchmarking/twins'):
    subset_list = ['500', '1000', '2000', '5000', 'None']
    for subset in subset_list:
        file_name = f"results_0.5_{subset}.csv"
        df = pd.read_csv(os.path.join(data_dir, file_name))

        TARNet_pehe = df['TARNet_pehe'].tolist()
        TARNet_single_pehe = df['TARNet_single_pehe'].tolist()
        TARNet_single_2_pehe = df['TARNet_single_2_pehe'].tolist()

        plt.figure(figsize=(10, 8))

        # scatter plots
        plt.scatter(range(5), TARNet_pehe, color='green', label='TARNet')
        plt.scatter(range(5), TARNet_single_pehe, color='purple', label='TARNet_single')
        plt.scatter(range(5), TARNet_single_2_pehe, color='orange', label='TARNet_single_2')

        # labels
        plt.xlabel('Variance')
        plt.ylabel('RMSE')
        plt.legend()

        if not os.path.exists(f"plots/twins/comparison/"):
            os.makedirs(f"plots/twins/comparison/")

        plt.savefig(f"plots/twins/comparison/twins_0.5_{subset}.png")

twins_plot()