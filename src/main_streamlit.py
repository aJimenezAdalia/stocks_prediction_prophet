'''
Streamlit main

'''

# Imports
import streamlit as st
import function_streamlit as ft
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Page settings
ft.config_page()

# Cache
st.cache(suppress_st_warning=True)

# Sidebar - Menu
menu = st.sidebar.selectbox('Menu', 
    ['Home', 'Validated Predictions', 'Future Forecast', 'Conclussions'])

if menu == 'Home':
    ft.home()
elif menu == 'Validated Predictions':
    ft.val_predictions()
elif menu == 'Future Forecast':
    ft.future()
elif menu == 'Conclussions':
    ft.conclussions()
