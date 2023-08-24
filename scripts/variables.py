# read/write helper functions for saving variables and ranges
import pandas as pd


def write_variables_to_csv(variables, filename):
    pd.DataFrame.from_dict(variables, orient="index").to_csv(filename, header=False)


def read_variables_from_csv(filename):
    return pd.read_csv(filename, index_col=0, header=None).T.to_dict(orient="list")
