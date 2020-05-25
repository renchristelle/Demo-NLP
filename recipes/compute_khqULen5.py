# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

import re
import os
import tarfile
import requests

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Output Folder
imdb_folder = dataiku.Folder("khqULen5")
dirname = imdb_folder.get_path()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Download archive
url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
response = requests.get(url)
imdb_folder.upload_stream("aclImdb_v1.tar.gz", response.content)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Untar archive into same directory
filename = os.path.join(dirname, "aclImdb_v1.tar.gz")
tar = tarfile.open(filename)
tar.extractall(path=dirname)
tar.close()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Remove archive
os.remove(filename)