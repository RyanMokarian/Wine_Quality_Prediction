from src.utils import read_data_to_dataframe
import pandas as pd



if __name__ == "__main__":
    df_red_wine =read_data_to_dataframe("..\data\winequality-red.csv")
    df_white_wine =read_data_to_dataframe("..\data\winequality-white.csv")
