#!/usr/bin/python
import pandas as pd
import os


p_ds = os.path.join(os.path.dirname(__file__), '../structural_rearrangement_data.csv')
df = pd.read_csv(p_ds)

