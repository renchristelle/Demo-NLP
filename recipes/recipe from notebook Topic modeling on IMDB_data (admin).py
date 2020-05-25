# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Topic Modeling on IMDB_data

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Topic models are statistical models that aim to discover the 'hidden' thematic structure in a collection of documents, i.e. identify possible topics in our corpus. It is an interative process by nature, as it is crucial to determine the right number of topics.
# 
# This notebook is organised as follows:
# 
# * [Setup and dataset loading](#setup)
# * [Text Processing:](#text_process) Before feeding the data to a machine learning model, we need to convert it into numerical features.
# * [Topics Extraction Models:](#mod) We present two differents models from the sklearn library: NMF and LDA.
# * [Topics Visualisation with pyLDAvis](#viz)
# * [Topics Clustering:](#clust)  We try to understand how topics relate to each other.
# * [Further steps](#next)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Setup and dataset loading <a id="setup" />
# 
# First of all, let's load the libraries that we'll use.
# 
# **This notebook requires the installation of the [pyLDAvis](https://pyldavis.readthedocs.io/en/latest/readme.html#installation) package.**
# [See here for help with intalling python packages.](https://www.dataiku.com/learn/guide/code/python/install-python-packages.html)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#%pylab inline
import warnings                         # Disable some warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import dataiku
from dataiku import pandasutils as pdu
import numpy as np, pandas as pd,  seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text

from sklearn.decomposition import LatentDirichletAllocation,NMF
import pyLDAvis.sklearn
#pyLDAvis.enable_notebook()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
dataset_limit = 10000

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# The first thing we do is now to load the dataset and identify possible text columns.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Take a handle on the dataset
mydataset = dataiku.Dataset("IMDB_data")

# Load the first lines.
# You can also load random samples, limit yourself to some columns, or only load
# data matching some filters.
#
# Please refer to the Dataiku Python API documentation for more information
df = mydataset.get_dataframe()#limit = dataset_limit)

df_orig = df.copy()

# Get the column names
numerical_columns = list(df.select_dtypes(include=[np.number]).columns)
categorical_columns = list(df.select_dtypes(include=[object]).columns)
date_columns = list(df.select_dtypes(include=['<M8[ns]']).columns)

# Print a quick summary of what we just loaded
print("Loaded dataset")
print("   Rows: %s" % df.shape[0])
print("   Columns: %s (%s num, %s cat, %s date)" % (df.shape[1],
                                                    len(numerical_columns), len(categorical_columns),
                                                    len(date_columns)))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# By default, we suppose that the text of interest for which we want to extract topics is the first of the categorical columns.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
raw_text_col = categorical_columns[0]

# Uncomment this if you want to take manual control over which variables is the text of interest
#print df.columns
#raw_text_col = "text_normalized"

raw_text = df[raw_text_col]
# Issue a warning if data contains NaNs
if(raw_text.isnull().any()):
    print('\x1b[33mWARNING: Your text contains NaNs\x1b[0m')
    print('Please take care of them, the countVextorizer will not be able to fit your data if it contains empty values.')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Text Processing <a id="text_process" />
# 
# We cannot directly feed the text to the Topics Extraction Algorithms. We first need to process the text in order to get numerical vectors. We achieve this by applying either a CountVectorizer() or a TfidfVectorizer(). For more information on those technics, please refer to thid [sklearn documentation](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html).

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# As with any text mining task, we first need to remove stop words that provide no useful information about topics. *sklearn* provides a default stop words list for english, but we can alway add to it any custom stop words : <a id="stop_words" /a>

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
custom_stop_words = ['movie', 'film']
#custom_stop_words = [u'did', u'good', u'right', u'said', u'does', u'way',u'edu', u'com', u'mail', u'thanks', u'post', u'address', u'university', u'email', u'soon', u'article',u'people', u'god', u'don', u'think', u'just', u'like', u'know', u'time', u'believe', u'say',u'don', u'just', u'think', u'probably', u'use', u'like', u'look', u'stuff', u'really', u'make', u'isn']

stop_words = text.ENGLISH_STOP_WORDS.union(custom_stop_words)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### CountVectorizer() on the text data <a id="tfidf" />
# 
# We first initialise a CountVectorizer() object and then apply the fit_transform method to the text.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
cnt_vectorizer = CountVectorizer(strip_accents = 'unicode',stop_words = stop_words,lowercase = True,
                                token_pattern = r'\b[a-zA-Z]{3,}\b', max_df = 0.85, min_df = 2)

text_cnt = cnt_vectorizer.fit_transform(raw_text)

print(text_cnt.shape)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### TfidfVectorizer() on the text data <a id="tfidf" />
# 
# We first initialise a TfidfVectorizer() object and then apply the fit_transform method to the text.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
tfidf_vectorizer = TfidfVectorizer(strip_accents = 'unicode',stop_words = stop_words,lowercase = True,
                                token_pattern = r'\b[a-zA-Z]{3,}\b', max_df = 0.75, min_df = 0.02)

text_tfidf = tfidf_vectorizer.fit_transform(raw_text)

print(text_tfidf.shape)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# In the following, we will apply the topics extraction to `text_tidf`.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Topics Extraction Models <a id="mod" />
# 
# There are two very popular models for topic modelling, both available in the sklearn library:
# 
# * [NMF (Non-negative Matrix Factorization)](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization),
# * [LDA (Latent Dirichlet Allocation)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)
# 
# Those two topic modeling algorithms infer topics from a collection of texts by viewing each document as a mixture of various topics. The only parameter we need to choose is the number of desired topics `n_topics`.
# It is recommended to try different values for `n_topics` in order to find the most insightful topics. For that, we will show below different analyses (most frequent words per topics and heatmaps).

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
n_topics= 5

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
topics_model = LatentDirichletAllocation(n_topics, random_state=0)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
topics_model.fit(text_tfidf)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Most Frequent Words per Topics
# An important way to assess the validity of our topic modelling is to directly look at the most frequent words in each topics.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Uncomment the following line to try NMF instead.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#topics_model = NMF(n_topics, random_state=0)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
n_top_words = 10
feature_names = tfidf_vectorizer.get_feature_names()

def get_top_words_topic(topic_idx):
    topic = topics_model.components_[topic_idx]

    print( [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]] )

for topic_idx, topic in enumerate(topics_model.components_):
    print ("Topic #%d:" % topic_idx )
    get_top_words_topic(topic_idx)
    print ("")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Pay attention to the words present, if some are very common you may want to go back to the [definition of custom stop words](#stop_words).

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### Naming the topics
# 
# Thanks to the above analysis, we can try to name each topics:

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
dict_topic_name = {i: "topic_"+str(i) for i in range(n_topics)}
dict_topic_name = {0: "Horror/Fantasy", 1:"Story description", 2:"Adaptation", 3:"Bad review", 4:"Good review"} #Define here your own name mapping and uncomment this !

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#dict_topic_name = {0: "Posting", 1: "Driving", 2: "OS (Windows)", 3: "Past", 4: "Games", 5: "Sales", 6: "Misc", 7: "Christianity", 8: "Personal information", 9: "Government/Justice"}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Topics Visualization with pyLDAvis <a id="viz">
# 
# Thanks to the pyLDAvis package, we can easily visualise and interpret the topics that has been fit to our corpus of text data.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
pyLDAvis.sklearn.prepare(topics_model, text_tfidf, tfidf_vectorizer)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Write Output

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import operator

# scoring the topics
topic_scores = pd.DataFrame(topics_model.transform(text_tfidf), columns = [dict_topic_name[i] for i in range(n_topics)])
topic_scores['final_topic'] = topic_scores.apply(lambda x:max(x.iteritems(), key=operator.itemgetter(1))[0], axis=1)

# Add scores to documents
df_with_topic_scored = pd.concat([df, topic_scores], axis=1)

# most relevant words per topic
top_word_per_topic_df = pd.DataFrame(columns = [dict_topic_name[i] for i in range(n_topics)])
for topic_idx, topic in enumerate(topics_model.components_):
    topic = topics_model.components_[topic_idx]
    top_word_per_topic_df[dict_topic_name[topic_idx]] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
reviews_with_topics = dataiku.Dataset("reviews_with_topics")
reviews_with_topics.write_with_schema(df_with_topic_scored)
top_words_per_topic = dataiku.Dataset("top_words_per_topic")
top_words_per_topic.write_with_schema(top_word_per_topic_df)