# %%
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle as pk
from urllib.request import urlopen
# %%
# DATA_URL = ('E:/')
os.chdir('E:/Projects/QuoraQuestionPairSimilarity/')
# %%
# @st.cache
def load_data():
    # model = pk.load(urlopen(DATA_URL, 'rb'))    
    df=pd.read_pickle("df_preprocessed.pkl")
    with open('Stack.pkl','rb') as pk_file:
        model = pk.load(pk_file)
    return df,model 

data_load_state =st.text('Loading Required dependencies...')

df,model = load_data()

data_load_state.text("Done!")

# %%
st.header("Quora Question Pair Similarity Prototype")
ques = st.text_area("",value= "Ask Question here...")

st.info('This is a purely informational message')
st.success('Similar question(s) Found!!')
