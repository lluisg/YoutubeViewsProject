#importing libraries
import csv
import os
import pandas as pd
import numpy as np
import json
from urllib.request import urlopen

if __name__ == "__main__":
    df = pd.read_csv('../DATA/videosID8M.csv', encoding = "utf-8")
    df.drop('pseudoId', inplace=True, axis=1)
    df.to_csv('../DATA/videosID8M_final.csv', encoding='utf-8', index=False)
