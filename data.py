#downloading and extracting the dataset on colab's server
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import tarfile

dest_directory = './data'
if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
filename = 'a.tar.gz'
filepath = os.path.join(dest_directory, filename)
if not os.path.exists(filepath):
    urllib.request.urlretrieve("https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz", filepath)
    tar = tarfile.open(filepath)
    print('a')
    tar.extractall(dest_directory)
    print('b')
    tar.close()



#forming a panda dataframe from the metadata file
data = pd.read_csv(os.path.join(dest_directory, "UrbanSound8K/metadata/UrbanSound8K.csv"))
#head of the dataframe
print(data.head())
