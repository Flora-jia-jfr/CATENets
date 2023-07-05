import pandas as pd
import matplotlib.pyplot as plt

def plot(setting="C", first_50=False):
    if setting == "C":
        # Read the CSV file for C setting
        df = pd.read_csv('results/experiments_benchmarking/ihdp/results_C.csv')
        Forest_df = pd.read_csv('results/experiments_benchmarking/ihdp/grf_original.csv')
        # Forest_df = Forest_df[Forest_df['simu'] <= 10]
        df['cf_out'] = Forest_df['cf_out']
        df['t_out'] = Forest_df['t_out']
        # df = df[df['cate_var_out'] <= 60]
    elif setting == "D":
        # Read the CSV file for D setting
        df = pd.read_csv('results/experiments_benchmarking/ihdp/results_D.csv')
        Forest_df = pd.read_csv('results/experiments_benchmarking/ihdp/grf_modified.csv')
        # Forest_df = Forest_df[Forest_df['simu'] <= 10]
        df['cf_out'] = Forest_df['cf_out']
        df['t_out'] = Forest_df['t_out']
        # df = df[df['cate_var_out'] <= 60]

    if first_50:
        # Sort by 'cate_var_out' and take top 50 rows
        df = df.sort_values(by='cate_var_out', ascending=True).head(50)

    # Get the lists
    cate_var_out = df['cate_var_out'].tolist()
    TNet_out = df['TNet_out'].tolist()
    TARNet_out = df['TARNet_out'].tolist()
    CF_out = df['cf_out'].tolist()
    TRF_out = df['t_out'].tolist()

    plt.figure(figsize=(10, 8))

    # scatter plots
    plt.scatter(cate_var_out, TNet_out, color='red', label='TNet')
    plt.scatter(cate_var_out, TARNet_out, color='green', label='TARNet')
    plt.scatter(cate_var_out, CF_out, color='orange', label='CF')
    plt.scatter(cate_var_out, TRF_out, color='blue', label='TRF')

    # labels
    plt.xlabel('Variance')
    plt.ylabel('RMSE')
    plt.legend()

    if first_50:
        plt.savefig(f"plots/IHDP_{setting}_first_50.png")
    else:
        plt.savefig(f"plots/IHDP_{setting}.png")

plot(setting="C", first_50=False)
plot(setting="C", first_50=True)
plot(setting="D", first_50=False)
plot(setting="D", first_50=True)