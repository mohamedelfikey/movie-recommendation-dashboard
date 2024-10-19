""" 
project name: shape calculations
author: Mohamed Ahmed Elfikey

"""

import streamlit as st
from model import top10 # variable
from model import titles
from model import get_recommendations

st.header("Movie Recomendations")

suggestion = titles
movie=st.selectbox("Enter movie name: ",titles)

recommendations=list (get_recommendations(movie).values)

search_button = st.button("search")
if search_button:
    st.write(recommendations)



st.sidebar.title("Most popular movies:")

with st.sidebar:
    st.text(f" 1- {top10.values[0]} \n 2- {top10.values[1]} \n 3- {top10.values[2]} \n 4- {top10.values[3]} \n 5- {top10.values[4]} \n 6- {top10.values[5]} \n 7- {top10.values[6]} \n 8- {top10.values[7]} \n 9- {top10.values[8]} \n 10- {top10.values[9]}   "  )