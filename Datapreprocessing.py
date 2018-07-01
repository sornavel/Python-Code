# Data Preprocessing# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
"""
Spyder Editor

This is a temporary script file.
"""

dataset = pd.read_csv('C:\Program Files\R\Datasets\Data_Preprocessing\Data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values
