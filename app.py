from matplotlib.pyplot import title
import streamlit as st
import pandas as pd
from PIL import Image
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


header = st.container()
dataset =st.container()

#load Images
@st.cache
def load_image(image_file):
    img =Image.open(image_file)
    return img

with header:
    st.title("Energy Disaggregation Analysis")
    image_file = st.file_uploader("Upload Building Image",type=["png","jpg","jpeg"])
    if image_file is not None:
        file_details = {"filename":image_file.name}
        st.write(file_details)
        st.image(load_image(image_file))
        st.write(type(image_file))

with dataset:
    st.header('Total Building Energy Data')
    data_Energy = st.file_uploader("Upload Total Energy Data in CSV format", type=["csv"])
    if data_Energy is not None:
        st.write(type(data_Energy))
        file_details = {"filename":data_Energy.name,"filetype":data_Energy.type,"filesize":data_Energy.size}
        st.write(file_details)
        df_energy = pd.read_csv(data_Energy,index_col=0)
        st.dataframe(df_energy)
        st.line_chart(df_energy['Total Energy (MJ)'])
################################################################
if st.button("Time Series Components"):
    stl_result = seasonal_decompose(df_energy['Total Energy (MJ)'], model='multiplicative',period=24*7)
    st.subheader('Trend')
    st.line_chart(stl_result.trend)
    st.subheader('Seasonality')
    st.line_chart(stl_result.seasonal)
    st.subheader('Residual')
    st.line_chart(stl_result.resid)
################################################################
if st.button("Time Series Decomposition"):
    stl_result = seasonal_decompose(df_energy['Total Energy (MJ)'], model='multiplicative',period=24*7)
############ Lighting and plugLoads ############################

    df_seasonal_elec = stl_result.seasonal*stl_result.trend.min()
    st.subheader('Lighting and plugLoads (MJ)')
    st.line_chart(df_seasonal_elec)

########### Cooling #############################################    
    bkp1 = '2018-04-28 08:00:00'
    bkp2 = '2018-09-24 08:00:00'
    df_clg = (stl_result.seasonal.loc[bkp1:bkp2]*stl_result.trend.loc[bkp1:bkp2]*stl_result.resid[bkp1:bkp2])
    df_clg = df_clg.loc[bkp1:bkp2] - df_seasonal_elec.loc[bkp1:bkp2]
    df_clg[df_clg < 0] = 0
    st.subheader('Cooling (MJ)')
    st.line_chart(df_clg)

########### Heating #############################################
    df_htg1 = (stl_result.seasonal.loc[:bkp1]*stl_result.trend.loc[:bkp1]*stl_result.resid.loc[:bkp1])
    df_htg2 = (stl_result.seasonal.loc[bkp2:]*stl_result.trend.loc[bkp2:]*stl_result.resid.loc[bkp2:])
    df_htg1 = df_htg1.loc[:bkp1]- df_seasonal_elec.loc[:bkp1]
    df_htg1[df_htg1 < 0 ] = 0
    df_htg2 = df_htg2.loc[bkp2:]- df_seasonal_elec.loc[bkp2:]
    df_htg2[df_htg2 < 0 ] = 0
    st.subheader('Heating  (MJ)')
    st.line_chart(df_htg1)
    st.line_chart(df_htg2)
#################################################################
    


   

