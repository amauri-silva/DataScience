import pandas as pd
import matplotlib as matplotlib
import numpy as np
import sys
sys.path.append("./config")
#from config import constants

from IPython.display import display

DATASET_CALENDAR = "calendar.csv"
DATASET_LISTING_TURISTICA = "listina_turistica.csv"
DATASET_LISTING = "listing.csv"
DATASET_REVIEWS = "reviews.csv"



def load_dataset(filename):
    return pd.read_csv("/home/amaurisilva/dev-environment/vscode-worspace/DataScience/AirBNB-RentalPriceModel/dataset/reviews.csv")

def return_dataframe(filename):
    csv_file = load_dataset(filename)
    return pd.DataFrame(csv_file)
    

if __name__ == "__main__":
    
    df = return_dataframe(DATASET_REVIEWS)
    display(df)
    #df.style