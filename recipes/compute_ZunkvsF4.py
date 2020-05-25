# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

import re
import os
import tarfile
import requests
import zipfile

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Output Folder
imdb_folder = dataiku.Folder("ZunkvsF4")
dirname = imdb_folder.get_path()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Download archive
url = "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip"
response = requests.get(url)
imdb_folder.upload_data("wiki.en.zip", response.content)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
filename = os.path.join(dirname, "wiki.en.zip")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Unzip archive into same directory
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall(dirname)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Remove archive
os.remove(filename)

# Remove text formats to only keep binary format
text_filename = os.path.join(dirname, "wiki.en.vec")
os.remove(text_filename)