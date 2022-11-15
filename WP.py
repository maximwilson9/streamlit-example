#Import packages
from collections import namedtuple
import altair as alt
import math
import numpy as np
import pandas as pd
import streamlit as st
import os

#


#Set Title
st.title('Winning Probability App for Hulu Media Tests')

"""
# Welcome to Streamlit!
Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:
Test line
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).
In the meantime, below is an example of what you can do with just a few lines of code:
"""
os.chdir('C:/Users/maxim.wilson/Documents/Projects/Facebook')
fb_excel = 'FY22Q4_TestNoTest.xlsx'
raw_fb = pd.read_excel(fb_excel,header=None,sheet_name='conversion metrics')
st.write(raw_fb)
