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

fb_excel = '2up_savings_messaging.csv'
raw_fb = pd.read.csv(fb_excel,header=None)
st.write(raw_fb)
