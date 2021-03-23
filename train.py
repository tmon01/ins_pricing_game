"""This script is used to train your model. You can modify it if you want."""

import numpy as np
import sys
import pandas as pd

# This script expects the dataset as a sys.args argument.
input_dataset = '../training.csv'  # The default value.
if len(sys.argv) >= 2:
	input_dataset = sys.argv[1]

# Load the dataset.
input_data = pd.read_csv(input_dataset)
Xraw = input_data.drop(columns=['claim_amount'])
yraw = input_data['claim_amount'].values


# Create a model, train it, then save it.
import model


new_model = model.fit_model(Xraw, yraw)

model.save_model(new_model)