from old.genetic_algo import GeneticAlgorithm
from datetime import datetime
import pandas as pd

def get_data(data_path: str, start_date: datetime, end_date: datetime):
    """
    Get the data to the right format

    :param data_path: path to the csv file with the WTI spot price data
    :param start_date: start of the time interval
    :param end_date: end of the time interval
    """

    df = pd.read_csv(data_path, sep=";")
    df["Date"] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df["Spot"] = pd.to_numeric(df["Spot"].str.replace(",", "."))

    df = df[(df["Date"] > start_date) & (df["Date"] < end_date)].reset_index(drop=True)
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days

    return df

df = get_data(data_path=r"C:\Users\thibc\OneDrive - De Vinci\Dauphine\Cours\Gestion Quant\Projet\Data.csv",
            start_date=datetime(month=4, day=1, year=2003), end_date=datetime(month=2, day=1, year=2008))

GA = GeneticAlgorithm()
GA.run(df)
print('end')