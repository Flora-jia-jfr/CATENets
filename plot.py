import pandas as pd
import os
# import matplotlib.pyplot as plt
import numpy as np

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


# # IHDP_plot(setting="D", first_50=False, dir_path='comparison_tarnet_single_2', forest=False)
# # IHDP_plot(setting="D", first_50=True, dir_path='comparison_tarnet_single_2', forest=False)
# IHDP_plot(setting="C", first_50=False, dir_path='comparison', forest=False)
# IHDP_plot(setting="C", first_50=True, dir_path='comparison', forest=False)
# IHDP_plot(setting="D", first_50=False, dir_path='comparison', forest=False)
# IHDP_plot(setting="D", first_50=True, dir_path='comparison', forest=False)
# # IHDP_plot(setting="D", first_50=False)
# # IHDP_plot(setting="D", first_50=True)
# ACIC2016_plot()
# ACIC2016_plot(first_50=True)
# twins_plot()


def IHDP_table_res(setting="C", dir_path=None):
    base_dir_path = 'results/experiments_benchmarking/ihdp'
    if setting == "C":
        if dir_path is not None:
            full_dir_path = os.path.join(base_dir_path, dir_path)
        else:
            full_dir_path = base_dir_path
        # Read the CSV file for C setting
        df = pd.read_csv(os.path.join(full_dir_path, 'results_C.csv'))
    elif setting == "D":
        if dir_path is not None:
            full_dir_path = os.path.join(base_dir_path, dir_path)
        else:
            full_dir_path = base_dir_path
        # Read the CSV file for D setting
        df = pd.read_csv(os.path.join(full_dir_path, 'results_D.csv'))

    # Get the lists
    cate_var_out = df['cate_var_out'].tolist()
    TARNet_out = df['TARNet_out'].tolist()
    TARNet_single_out = df['TARNet_single_out'].tolist()
    TARNet_single_2_out = df['TARNet_single_2_out'].tolist()
    TARNet_ate_out = df['TARNet_ate_out'].tolist()
    TARNet_single_ate_out = df['TARNet_single_ate_out'].tolist()
    TARNet_single_2_ate_out = df['TARNet_single_2_ate_out'].tolist()

    # Get mean and standard deviations from the lists
    TARNet_out_mean = np.mean(TARNet_out)
    TARNet_out_std = np.std(TARNet_out)
    TARNet_ate_out_mean = np.mean(TARNet_ate_out)
    TARNet_ate_out_std = np.std(TARNet_ate_out)

    TARNet_single_out_mean = np.mean(TARNet_single_out)
    TARNet_single_out_std = np.std(TARNet_single_out)
    TARNet_single_ate_out_mean = np.mean(TARNet_single_ate_out)
    TARNet_single_ate_out_std = np.std(TARNet_single_ate_out)

    TARNet_single_2_out_mean = np.mean(TARNet_single_2_out)
    TARNet_single_2_out_std = np.std(TARNet_single_2_out)
    TARNet_single_2_ate_out_mean = np.mean(TARNet_single_2_ate_out)
    TARNet_single_2_ate_out_std = np.std(TARNet_single_2_ate_out)

    # Prepare the data
    data = {
        'RMSE': [
            f"{TARNet_out_mean:.2f} \u00B1 {TARNet_out_std:.2f}",
            f"{TARNet_single_out_mean:.2f} \u00B1 {TARNet_single_out_std:.2f}",
            f"{TARNet_single_2_out_mean:.2f} \u00B1 {TARNet_single_2_out_std:.2f}"
        ],
        'ATE_error': [
            f"{TARNet_ate_out_mean:.2f} \u00B1 {TARNet_ate_out_std:.2f}",
            f"{TARNet_single_ate_out_mean:.2f} \u00B1 {TARNet_single_ate_out_std:.2f}",
            f"{TARNet_single_2_ate_out_mean:.2f} \u00B1 {TARNet_single_2_ate_out_std:.2f}"
        ]
    }

    # Create the DataFrame
    df = pd.DataFrame(data, index=['TARNet', 'TARNet_single', 'TARNet_single_2'])

    # Display the DataFrame
    print(df)

    df.to_csv(f"plots/{dir_path}/{setting}_table.csv")

# IHDP_table_res(setting="C", dir_path='comparison')
# IHDP_table_res(setting="D", dir_path='comparison')

def ACIC2016_table_res(data_dir='results/experiments_benchmarking/acic2016/comparsion', first_50=False):
    df = pd.read_csv(os.path.join(data_dir, 'results_False_2_4000.csv'))

    # Get the lists
    cate_var_out = df['cate_var_out'].tolist()
    TARNet_out = df['TARNet_out'].tolist()
    TARNet_single_out = df['TARNet_single_out'].tolist()
    TARNet_single_2_out = df['TARNet_single_2_out'].tolist()
    TARNet_ate_out = df['TARNet_ate_out'].tolist()
    TARNet_single_ate_out = df['TARNet_single_ate_out'].tolist()
    TARNet_single_2_ate_out = df['TARNet_single_2_ate_out'].tolist()

    # Get mean and standard deviations from the lists
    TARNet_out_mean = np.mean(TARNet_out)
    TARNet_out_std = np.std(TARNet_out)
    TARNet_ate_out_mean = np.mean(TARNet_ate_out)
    TARNet_ate_out_std = np.std(TARNet_ate_out)

    TARNet_single_out_mean = np.mean(TARNet_single_out)
    TARNet_single_out_std = np.std(TARNet_single_out)
    TARNet_single_ate_out_mean = np.mean(TARNet_single_ate_out)
    TARNet_single_ate_out_std = np.std(TARNet_single_ate_out)

    TARNet_single_2_out_mean = np.mean(TARNet_single_2_out)
    TARNet_single_2_out_std = np.std(TARNet_single_2_out)
    TARNet_single_2_ate_out_mean = np.mean(TARNet_single_2_ate_out)
    TARNet_single_2_ate_out_std = np.std(TARNet_single_2_ate_out)

    # Prepare the data
    data = {
        'RMSE': [
            f"{TARNet_out_mean:.2f} \u00B1 {TARNet_out_std:.2f}",
            f"{TARNet_single_out_mean:.2f} \u00B1 {TARNet_single_out_std:.2f}",
            f"{TARNet_single_2_out_mean:.2f} \u00B1 {TARNet_single_2_out_std:.2f}"
        ],
        'ATE_error': [
            f"{TARNet_ate_out_mean:.2f} \u00B1 {TARNet_ate_out_std:.2f}",
            f"{TARNet_single_ate_out_mean:.2f} \u00B1 {TARNet_single_ate_out_std:.2f}",
            f"{TARNet_single_2_ate_out_mean:.2f} \u00B1 {TARNet_single_2_ate_out_std:.2f}"
        ]
    }

    # Create the DataFrame
    df = pd.DataFrame(data, index=['TARNet', 'TARNet_single', 'TARNet_single_2'])

    # Display the DataFrame
    print(df)

    df.to_csv(f"plots/ACIC_2016/table.csv")
    
ACIC2016_table_res()

def twins_table_res(data_dir='results/experiments_benchmarking/twins'):
    subset_list = ['500', '1000', '2000', '5000', 'None']
    for subset in subset_list:
        file_name = f"results_0.5_{subset}.csv"
        df = pd.read_csv(os.path.join(data_dir, file_name))

        TARNet_pehe = df['TARNet_pehe'].tolist()
        TARNet_single_pehe = df['TARNet_single_pehe'].tolist()
        TARNet_single_2_pehe = df['TARNet_single_2_pehe'].tolist()
        TARNet_ate = df['TARNet_ate'].tolist()
        TARNet_single_ate = df['TARNet_single_ate'].tolist()
        TARNet_single_2_ate = df['TARNet_single_2_ate'].tolist()

        # Get mean and standard deviations from the lists
        TARNet_out_mean = np.mean(TARNet_pehe)
        TARNet_out_std = np.std(TARNet_pehe)
        TARNet_ate_out_mean = np.mean(TARNet_ate)
        TARNet_ate_out_std = np.std(TARNet_ate)

        TARNet_single_out_mean = np.mean(TARNet_single_pehe)
        TARNet_single_out_std = np.std(TARNet_single_pehe)
        TARNet_single_ate_out_mean = np.mean(TARNet_single_ate)
        TARNet_single_ate_out_std = np.std(TARNet_single_ate)

        TARNet_single_2_out_mean = np.mean(TARNet_single_2_pehe)
        TARNet_single_2_out_std = np.std(TARNet_single_2_pehe)
        TARNet_single_2_ate_out_mean = np.mean(TARNet_single_2_ate)
        TARNet_single_2_ate_out_std = np.std(TARNet_single_2_ate)

        # Prepare the data
        data = {
            'PEHE': [
                f"{TARNet_out_mean:.2f} \u00B1 {TARNet_out_std:.2f}",
                f"{TARNet_single_out_mean:.2f} \u00B1 {TARNet_single_out_std:.2f}",
                f"{TARNet_single_2_out_mean:.2f} \u00B1 {TARNet_single_2_out_std:.2f}"
            ],
            'ATE_error': [
                f"{TARNet_ate_out_mean:.2f} \u00B1 {TARNet_ate_out_std:.2f}",
                f"{TARNet_single_ate_out_mean:.2f} \u00B1 {TARNet_single_ate_out_std:.2f}",
                f"{TARNet_single_2_ate_out_mean:.2f} \u00B1 {TARNet_single_2_ate_out_std:.2f}"
            ]
        }

        # Create the DataFrame
        df = pd.DataFrame(data, index=['TARNet', 'TARNet_single', 'TARNet_single_2'])

        # Display the DataFrame
        print(df)
        df.to_csv(f"plots/twins/{subset}_table.csv")
        
# twins_table_res()

def speedDating_table_res(data_dir='results/experiments_benchmarking/speedDating/comparison'):
    df = pd.read_csv(os.path.join(data_dir, 'speedDating.csv'))
    model_columns_out = ['TARNet_out', 'TARNet_single_out', 'TARNet_single_2_out']
    # Group by 'mod' and 'dim' and calculate mean and std for the model columns
    grouped_out = df.groupby(['mod', 'dim'])[model_columns_out].agg(['mean', 'std'])
    # Create a multi-index using the model types
    multi_index_out = pd.MultiIndex.from_product([['low', 'med', 'high'], 
                                                ['TARNet', 'TARNet_single', 'TARNet_single_2']],
                                                names=['dim', 'model'])
    # Initialize an empty dataframe with the desired multi-index and columns
    result_df_out = pd.DataFrame(index=multi_index_out, columns=[1, 2, 3, 4])
    # Populate the dataframe with mean ± std values
    for mod in [1, 2, 3, 4]:
        for dim in ['low', 'med', 'high']:
            for model in ['TARNet', 'TARNet_single', 'TARNet_single_2']:
                mean = grouped_out.loc[(mod, dim), (f'{model}_out', 'mean')]
                std = grouped_out.loc[(mod, dim), (f'{model}_out', 'std')]
                result_df_out.loc[(dim, model), mod] = f"{mean:.4f} ± {std:.4f}"
    print(result_df_out)

    result_df_out.to_csv(f'plots/speedDating/speedDating_table.csv')

# speedDating_table_res()

