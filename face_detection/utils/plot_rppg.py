import pandas as pd
import matplotlib.pyplot as plt

def plot_rppg_from_cvs(file_path):
    df = pd.read_csv(file_path)

    for column in df.columns:
        # print(df[column])
        plt.figure()
        plt.title(column)
        plt.plot(df[column])
        plt.show()

if __name__ == '__main__':

    file_path = "/Volumes/externiMAC/UBFC-Phys/dataset/s1/rppg/rppg_all_s1_T1.csv"
    plot_rppg_from_cvs(file_path)