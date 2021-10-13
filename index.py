# %%
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle as pk
from urllib.request import urlopen
# %%
DATA_URL = ('')

# %%
@st.cache
def load_data():
    model = pk.load(urlopen(DATA_URL, 'rb'))
    return model 

data_load_state =st.text('Loading Required dependencies...')

model = load_data()

data_load_state.text("Done!")

