import pandas as pd
import numpy as np
import pickle
import json

with open(r"C:\SJSU\Spring 2025\DATA 230\002 Project\HMDA\hmda-loan-predictor\app\county_census_data.json", "r") as json_file:
    county_census_data = json.load(json_file)


def prepare_data(sample_data):
    county_info = sample_data["county_name"]+"_"+str(sample_data["census_tract_number"])

    # Retrieve info
    retrieved_data = county_census_data[county_info]

    sample_data.update(retrieved_data)

    return sample_data