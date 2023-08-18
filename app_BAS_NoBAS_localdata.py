# Imports from standard libraries
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import time
import altair as alt
import seaborn as sns
from scipy.stats import pearsonr
import requests
import json
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from API_puller import API_puller
from datetime import datetime, date, time
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import mpld3
import streamlit.components.v1 as components
import random
import plotly.graph_objects as go


# Set the path to your logo file
logo_path = "logo.png"

# Add a title and logo to the app
st.set_page_config(page_title="My App", page_icon=logo_path)
# from sklearn.linear_model import LinearRegression
original_title = '<p style="font-family:Courier; color:Blue; font-size: 20px;">Original image</p>'
tab1, tab2,tab3 = st.tabs([ "Disaggregation with BAS data (Manual Upload)", "Disaggregation without BAS","Disaggregation with BAS data (API)" ])

with tab1:
    st.title('Energy Disaggregation using BAS Trend Data')

    # col1, col2, col3 = st.columns(3)

    # with col1:
    header = st.container()
    dataset =st.container()
    # progress_bar = st.sidebar.progress(0)
    # status_text = st.sidebar.empty()

    # @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    st.title("Cooling Data")
    # data_Energy = st.file_uploader("Upload Total Energy Data in CSV format", type=["csv"])
    #     if data_Energy is not None:
    #         st.write(type(data_Energy))
    #         file_details = {"filename":data_Energy.name,"filetype":data_Energy.type,"filesize":data_Energy.size}
    #         st.write(file_details)
    #         df_energy = pd.read_csv(data_Energy,index_col=0)
    #         # df_energy.index= pd.to_datetime(df_energy.index)
    #         st.write(file_details)
    #         st.dataframe(df_energy)
    uploaded_file_Clg = st.file_uploader("Upload Cooling Energy Data and BAS in CSV format",type=["csv"])
    if uploaded_file_Clg is not None:
        # df_Clg = pd.read_csv('Trend Logs (3).csv',index_col=0)
        df_Clg = pd.read_csv(uploaded_file_Clg,index_col=0)
        df_Clg.index = pd.to_datetime(df_Clg.index)
        df_Clg_2 = df_Clg.fillna(method="Pad")
        df_Clg_3 = df_Clg_2.resample('H').mean()
    # plt.plot(df_Clg_2['Jean Talon- Chilled Water']) 
        print(df_Clg_3)
        st.write(df_Clg_3)
        
        # # building_number = st.multiselect('Insert building number',[3642,3035])
        # building_number = 3642 # Building
        # st.write('The building number is ', building_number)

    # st.header("Date Range")
    # # start_date = st.date_input('start date',min_value=datetime.date(2017, 6, 10), max_value=datetime.date(2019, 1, 1))
    # min_date = date(2017, 6, 10)
    # max_date = date(2019, 1, 1)
    # a_date = st.date_input("Pick a date",  min_value=min_date,max_value=max_date)

    # min_value=datetime.date(2021, 1, 1),
        # max_value=datetime.date(2023, 1, 1),
    # end_date = st.date_input('end date',min_value=datetime.date(2017, 6, 10), max_value=datetime.date(2019, 1, 1))
    # print(start_date)
    # print(end_date)
    ######################
    if st.button('Cooling disaggrgation Results'):
        st.header('Cooling Valves (%)')
        df_valves = df_Clg_3.drop(['Jean Talon- Chilled Water (MJ)'], axis=1)
        # line_chart = alt.Chart(df_valves).mark_line(interpolate='basis').properties(title='Sales of consumer goods')
        # st.altair_chart(line_chart)
        st.line_chart(df_valves)
        # print(df_Htg['Jean Talon- Steam'].iloc[1:1000])
        st.header('Cooling Energy Meter (MJ)')
        st.line_chart(data = df_Clg_3,y='Jean Talon- Chilled Water (MJ)')
        # st.line_chart(data = df_Clg_3, x=df_Clg_3.index , y = "temp_max")
        # df_Clg_3.corr(numeric_only=True)
        # fig = plt.figure() 
        # plt.plot(df_Clg['Jean Talon- Chilled Water']) 

        # st.pyplot(fig)
    
        ############################ Weather Data #######################

        st.header('Outdoor Air Temperature (degC)')
        # Available time formats: LST/UTC
        timeformat = 'LST'

        # check https://power.larc.nasa.gov/#resources for full list of parameters and modify as needed

        # ALLSKY_SFC_SW_DNI - direct normal irradiance (W/m2)
        # ALLSKY_SFC_SW_DIFF - diffuse horizontal irradiance (W/m2)
        # T2M - temperature 2 m above ground (degC)
        # RH2M - relative humidity 2 m above ground level (m/s)
        # WS2M - wind speed 2 m above ground level (m/s)

        # params = 'ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF,T2M,RH2M,WS2M'
        params = 'T2M'
        #Always use RE (renewable energy) for this purpose
        community = 'RE' 
        #Obtain LAT/LON from google maps
        location = {
            'latitude':'45.73906',
            'longitude':'-75.73906'
            }
        # Start/end time in format: 'YYYYMMDD'
        

        sTime = '20170609'

        eTime = '20190101'
        # sTime = str(df_Clg.index[0])
        # sTime = sTime.replace('-','')
        # eTime = str(df_Clg.index[-1])
        # eTime = eTime.replace('-','')
        # print(eTime)

        # %% API call for given lat/long
        cwd = Path.cwd()
        path = cwd.__str__()
        url = 'https://power.larc.nasa.gov/api/temporal/hourly/point?Time='+timeformat+'&parameters='+params+'&community='+community+'&longitude='+location['longitude']+'&latitude='+location['latitude']+'&start='+sTime+'&end='+eTime+'&format=JSON'
        data = requests.get(url)

        data = data.json()
        data = pd.DataFrame((data['properties']['parameter']))
        data = data.set_index(pd.to_datetime(data.index, format='%Y%m%d%H'))

        st.line_chart(data,y='T2M')
        print(data)  

    ############Cooling regression Model###########################
        st.header('Cooling Energy Use by Each AHU (MJ)')

        df_model_cooling = pd.merge(df_Clg_3, data['T2M'], left_index=True, right_index=True)
        # df_model_cooling = pd.merge(df_model_cooling,df, left_index=True, right_index=True)
        predictors = df_model_cooling.drop(['Jean Talon- Chilled Water (MJ)'], axis=1)
        response = df_model_cooling['Jean Talon- Chilled Water (MJ)']
        print('df_model_cooling =')
        print(df_model_cooling)
        print(predictors.shape[1])
        print('Predictors =')
        print(predictors)
        # print(response)
        # print(data['T2M'])
        def rmse_ClgMdl(x): #Heating disagg model
            h = 0
            for i in range(predictors.shape[1]-1):
                h = x[i]*predictors.iloc[:,i] + h

            h = h + x[i+1]*(predictors.iloc[:,i+1]-x[i+2])*(predictors.iloc[:,i+1]-x[i+2]) + x[i+3]
            return np.sqrt(((response - h) ** 2).mean())

        x0 = np.zeros(predictors.shape[1]+2)
        b = (0.01,100)
        bnds = (b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b)
        solution_Clg = minimize(rmse_ClgMdl,x0,method='SLSQP',bounds=bnds)
        print(solution_Clg)
        print(predictors.columns)
        cooling_ahu = pd.DataFrame()

        # for i in range(0,predictors.shape[1]):
        #     heating_ahu = pd.concat([heating_ahu,(predictors.iloc[:,i]*solution_Htg[i])], axis=1)

        # heating_perimeter = solution_Htg[-2] * (predictors.iloc[:,-1]-solution_Htg[-2])*(predictors.iloc[:,-1]-solution_Htg[-2]) + solution_Htg[-1] *np.ones(df_model['AH7_HeatingValve'].shape)
        # labels = trend_log_data.columns
        # fig, axs = plt.subplots(14,sharex=True, figsize=(15,30))
        # for i in range(0,14):
        #     fig.suptitle('Heating Disaggregation ')
        #     axs[i].plot(solution_Htg.x[i]*predictors.iloc[:,i], color='red',linewidth=1, linestyle='dashed')
        #     axs[i].set(ylabel='kW')
        #     axs[i].set_title(labels[i])
        #   # axs[i].set_xlim(2000,2400)
        #     fig.tight_layout(pad=2)
        #     fig.subplots_adjust(top=0.9)
        # st.plotly_chart(fig)

        df_Clg_ahu = pd.DataFrame(columns = df_valves.columns)

        for i in range(predictors.shape[1]-1):
            this_column = df_Clg_ahu.columns[i]
            df_Clg_ahu[this_column] = solution_Clg.x[i]*predictors.iloc[:,i]

        print(df_Clg_ahu)

        cooling_disagg = pd.DataFrame()
        cooling_perimeter_other = solution_Clg.x[-3]*(predictors.iloc[:,-1]-solution_Clg.x[-2])*(predictors.iloc[:,-1]-solution_Clg.x[-2]) + solution_Clg.x[-1]
        print(cooling_perimeter_other)

        cooling_disagg = pd.DataFrame()
        cooling_disagg = pd.merge(df_Clg_ahu,cooling_perimeter_other, left_index=True, right_index=True)
        cooling_disagg['Perimeter Heaters and Others'] = cooling_disagg['T2M']
        cooling_disagg = cooling_disagg.drop(['T2M'], axis=1)
        BLDGE_AREA =70,970
        print(cooling_disagg)

        kpi_clg_ahu = df_Clg_ahu.sum()
        
        kpi_clg_perimeter_other = cooling_perimeter_other.sum()
        kpi_clg = pd.DataFrame({"category":[kpi_clg_ahu.index],"value":[kpi_clg_ahu.values]})
        print(kpi_clg)
        # fig = plt.figure(figsize = (10, 5))
        # merged_list = kpi_clg_ahu+cooling_perimeter_other
        # creating the bar plot
        # plt.bar(m.index, merged_list.values,
        #         width = 0.4)
        ########### Bar chart ########################
        # print(kpi_clg_ahu)
        # print(kpi_clg_ahu.values)
        # # "Energy Costs By Month"
        source = pd.DataFrame()
        source.index = kpi_clg_ahu.index
        source['AHU Cooling Energy(MJ)'] =kpi_clg_ahu.values
        st.bar_chart(source)
        # st.bar_chart(kpi_clg_ahu)
        # data
        #################################################
    #     label = ["Cooling Energy Use", 'AH10', 'AH14', 'AH2',
    #    'AH8', 'AH9', 'Pent_Elev',
    #    'AH7', 'AH16', 'AH11',
    #    'AH4', 'AH15', 'AH5',
    #    'AH12', 'AH13']
    #     source = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     target = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    #     value = kpi_clg_ahu.values
    #     # data to dict, dict to sankey
    #     link = dict(source = source, target = target, value = value)
    #     node = dict(label = label, pad=50, thickness=7)
    #     data = go.Sankey(link = link, node=node)
    #     # plot
    #     fig = go.Figure(data)
    #     fig

        # base = alt.Chart(kpi_clg_ahu).encode(alt.Theta("value:Q").stack(True),alt.Color("category:N").legend(None))
        # pie = base.mark_arc(outerRadius=120)
        # text = base.mark_text(radius=140, size=20).encode(text="category:N")
        # st.altair_chart(base)
        # params = {"ytick.color" : "w",
        #   "xtick.color" : "w",
        #   "axes.labelcolor" : "w",
        #   "axes.edgecolor" : "w"}
        # plt.rcParams.update(params)
        # explode = (0, 0.1, 0, 0,0,0,0.2,0,0,0,0,0,0,0)  # only "explode" the 2nd slice (i.e. 'Hogs')
        fig1, ax1 = plt.subplots()
        # ax1.pie(kpi_clg_ahu.values, explode=explode, labels=kpi_clg_ahu.index, autopct='%1.1f%%',
        # shadow=False, startangle=45)
        # ax1.axis('equal') 
        # fig1.patch.set_facecolor('black') # Equal aspect ratio ensures that pie is drawn as a circle.
        # fig1.patch.set_ec('white')
        # st.pyplot(fig1)

        # patches, texts, pcts = ax1.pie(
        #     kpi_clg_ahu.values, labels=kpi_clg_ahu.index, autopct='%.1f%%',
        #     wedgeprops={'linewidth': 1.0, 'edgecolor': 'white'},
        #     textprops={'size': 'x-large'},
        #     startangle=45)
        # # For each wedge, set the corresponding text label color to the wedge's
        # # face color.
        # for i, patch in enumerate(patches):
        #     texts[i].set_color(patch.get_facecolor())
        #     plt.setp(pcts, color='white')
        # plt.setp(texts)
        # ax1.set_title('Cooling energy use by AHUs %')
        # plt.tight_layout()
        # fig1.patch.set_facecolor('black')
        label_ahu=['AH10', 'AH14', 'AH2',
       'AH8', 'AH9', 'Pent_Elev',
       'AH7', 'AH16', 'AH11',
       'AH4', 'AH15', 'AH5',
       'AH12', 'AH13']
        colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        for i in range(14)]
        patches, texts = plt.pie(kpi_clg_ahu.values,colors=colors,startangle=90, labels=label_ahu)

        labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(label_ahu,100.*kpi_clg_ahu.values/kpi_clg_ahu.values.sum())]
        plt.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1), fontsize=6)
        # fig1.patch.set_facecolor('black')
        plt.ylabel("")
        st.pyplot(fig1)


       
        
        # bar_chart = alt.Chart(source).mark_bar().encode(y='Cooling Energy (MJ):Q',x='AHU:O')
        # st.altair_chart(bar_chart, use_container_width=True)


        # fig_html = mpld3.fig_to_html(fig1)
        # components.html(fig_html, height=700)
        
        # st.vega_lite_chart(kpi_clg_ahu)
        # plt.xlabel("AHU")
        # plt.ylabel("Total Energy Use(MJ)")
        # plt.title("Cooling Energy Use by AHUs")
        # plt.xticks(rotation = 90)
        # plt.show()
    # st.title("Heating Data")
    # uploaded_file_Htg = st.file_uploader("Upload Heating Energy Data and BAS in CSV format",type=["csv"])
    # if uploaded_file_Htg is not None:
    #     # df_Clg = pd.read_csv('Trend Logs (3).csv',index_col=0)
    #     df_Htg = pd.read_csv(uploaded_file_Htg,index_col=0)
    #     df_Htg.index = pd.to_datetime(df_Htg.index)
    #     df_Htg_2 = df_Htg.fillna(method="Pad")
    #     df_Htg_3 = df_Htg_2.resample('H').mean()
    # # plt.plot(df_Clg_2['Jean Talon- Chilled Water']) 
    #     print(df_Htg_3)
    #     st.write(df_Htg_3)
    # if st.button('Heating disaggrgation Results'):
    #     st.header('Heating Valves')
    #     df_valves_htg = df_Htg_3.drop(['Jean Talon- Steam (MJ)'], axis=1)
    #     # line_chart = alt.Chart(df_valves).mark_line(interpolate='basis').properties(title='Sales of consumer goods')
    #     # st.altair_chart(line_chart)
    #     st.line_chart(df_valves_htg)
    #     # print(df_Htg['Jean Talon- Steam'].iloc[1:1000])
    #     st.header('Cooling Energy Meter')
    #     st.line_chart(data = df_Htg_3,y='Jean Talon- Steam (MJ)')
    #     # st.line_chart(data = df_Clg_3, x=df_Clg_3.index , y = "temp_max")
    #     # df_Clg_3.corr(numeric_only=True)
    #     # fig = plt.figure() 
    #     # plt.plot(df_Clg['Jean Talon- Chilled Water']) 

    #     # st.pyplot(fig)
    
    #     ############################ Weather Data #######################

    #     st.header('Outdoor Air Temperature')
    #     # Available time formats: LST/UTC
    #     timeformat = 'LST'

    #     # check https://power.larc.nasa.gov/#resources for full list of parameters and modify as needed

    #     # ALLSKY_SFC_SW_DNI - direct normal irradiance (W/m2)
    #     # ALLSKY_SFC_SW_DIFF - diffuse horizontal irradiance (W/m2)
    #     # T2M - temperature 2 m above ground (degC)
    #     # RH2M - relative humidity 2 m above ground level (m/s)
    #     # WS2M - wind speed 2 m above ground level (m/s)

    #     # params = 'ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF,T2M,RH2M,WS2M'
    #     params = 'T2M'
    #     #Always use RE (renewable energy) for this purpose
    #     community = 'RE' 
    #     #Obtain LAT/LON from google maps
    #     location = {
    #         'latitude':'45.73906',
    #         'longitude':'-75.73906'
    #         }
    #     # Start/end time in format: 'YYYYMMDD'
        

    #     sTime = '20180101'

    #     eTime = '20181231'
    #     # sTime = str(df_Clg.index[0])
    #     # sTime = sTime.replace('-','')
    #     # eTime = str(df_Clg.index[-1])
    #     # eTime = eTime.replace('-','')
    #     # print(eTime)

    #     # %% API call for given lat/long
    #     cwd = Path.cwd()
    #     path = cwd.__str__()
    #     url = 'https://power.larc.nasa.gov/api/temporal/hourly/point?Time='+timeformat+'&parameters='+params+'&community='+community+'&longitude='+location['longitude']+'&latitude='+location['latitude']+'&start='+sTime+'&end='+eTime+'&format=JSON'
    #     data = requests.get(url)

    #     data = data.json()
    #     data = pd.DataFrame((data['properties']['parameter']))
    #     data = data.set_index(pd.to_datetime(data.index, format='%Y%m%d%H'))

    #     st.line_chart(data,y='T2M')
    #     print(data)  

    # ############Cooling regression Model###########################
    #     st.header('Heating Energy Use by Each AHU')

    #     df_model_Heating = pd.merge(df_Htg_3, data['T2M'], left_index=True, right_index=True)
    #     # df_model_cooling = pd.merge(df_model_cooling,df, left_index=True, right_index=True)
    #     predictors = df_model_Heating.drop(['Jean Talon- Steam (MJ)'], axis=1)
    #     response = df_model_Heating['Jean Talon- Steam (MJ)']
    #     print('df_model_Heating =')
    #     print(df_model_Heating)
    #     print(predictors.shape[1])
    #     print('Predictors =')
    #     print(predictors)
    #     # print(response)
    #     # print(data['T2M'])
    #     def rmse_HtgMdl(x): #Heating disagg model
    #         h = 0
    #         for i in range(predictors.shape[1]-1):
    #             h = x[i]*predictors.iloc[:,i] + h

    #         h = h + x[i+1]*(predictors.iloc[:,i+1]-x[i+2])*(predictors.iloc[:,i+1]-x[i+2]) + x[i+3]
    #         return np.sqrt(((response - h) ** 2).mean())

    #     x0 = np.zeros(predictors.shape[1]+2)
    #     b = (0.0,100)
    #     bnds = (b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b)
    #     solution_Htg = minimize(rmse_HtgMdl,x0,method='SLSQP',bounds=bnds)
    #     print(solution_Htg)
    #     print(predictors.columns)
    #     Heating_ahu = pd.DataFrame()

    #     # for i in range(0,predictors.shape[1]):
    #     #     heating_ahu = pd.concat([heating_ahu,(predictors.iloc[:,i]*solution_Htg[i])], axis=1)

    #     # heating_perimeter = solution_Htg[-2] * (predictors.iloc[:,-1]-solution_Htg[-2])*(predictors.iloc[:,-1]-solution_Htg[-2]) + solution_Htg[-1] *np.ones(df_model['AH7_HeatingValve'].shape)
    #     # labels = trend_log_data.columns
    #     # fig, axs = plt.subplots(14,sharex=True, figsize=(15,30))
    #     # for i in range(0,14):
    #     #     fig.suptitle('Heating Disaggregation ')
    #     #     axs[i].plot(solution_Htg.x[i]*predictors.iloc[:,i], color='red',linewidth=1, linestyle='dashed')
    #     #     axs[i].set(ylabel='kW')
    #     #     axs[i].set_title(labels[i])
    #     #   # axs[i].set_xlim(2000,2400)
    #     #     fig.tight_layout(pad=2)
    #     #     fig.subplots_adjust(top=0.9)
    #     # st.plotly_chart(fig)

    #     df_Htg_ahu = pd.DataFrame(columns = df_valves_htg.columns)

    #     for i in range(predictors.shape[1]-1):
    #         this_column = df_Htg_ahu.columns[i]
    #         df_Htg_ahu[this_column] = solution_Htg.x[i]*predictors.iloc[:,i]

    #     print(df_Htg_ahu)


    #     Heating_disagg = pd.DataFrame()
    #     Heating_perimeter_other = solution_Htg.x[-3]*(predictors.iloc[:,-1]-solution_Htg.x[-2])*(predictors.iloc[:,-1]-solution_Htg.x[-2]) + solution_Htg.x[-1]
    #     print(Heating_perimeter_other)

    #     Heating_disagg = pd.DataFrame()
    #     Heating_disagg = pd.merge(df_Htg_ahu,Heating_perimeter_other, left_index=True, right_index=True)
    #     Heating_disagg['Perimeter Heaters and Others'] = Heating_disagg['T2M']
    #     Heating_disagg = Heating_disagg.drop(['T2M'], axis=1)
    #     BLDGE_AREA =70,970
    #     print(Heating_disagg)

    #     kpi_Htg_ahu = df_Htg_ahu.sum()
        
    #     kpi_Htg_perimeter_other = Heating_perimeter_other.sum()
    #     kpi_Htg = pd.DataFrame({"category":[kpi_Htg_ahu.index],"value":[kpi_Htg_ahu.values]})
    #     print(kpi_Htg)
    #     fig = plt.figure(figsize = (10, 5))
    #     # merged_list = kpi_clg_ahu+cooling_perimeter_other
    #     # creating the bar plot
    #     # plt.bar(m.index, merged_list.values,
    #     #         width = 0.4)
    #     print(kpi_Htg_ahu)
    #     print(kpi_Htg_ahu.values)
    #     st.bar_chart(kpi_Htg_ahu)
    #     fig2, ax2 = plt.subplots()
    #     colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    #     for i in range(14)]
    #     patches, texts = plt.pie(kpi_Htg_ahu.values,colors=colors,startangle=90, labels=kpi_Htg_ahu.index)

    #     labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(kpi_Htg_ahu.index,100.*kpi_Htg_ahu.values/kpi_Htg_ahu.values.sum())]
    #     plt.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1), fontsize=8)
    #     # fig1.patch.set_facecolor('black')
    #     plt.ylabel("")
    #     st.pyplot(fig2)


with tab2:
    header = st.container()
    dataset =st.container()

    st.write("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fascinate');
    html, body, [class*="css"]  {
       font-family: 'sans serif', cursive;
    }
    </style>
    """, unsafe_allow_html=True)

    #load Images
    @st.cache
    def load_image(image_file):
        img =Image.open(image_file)
        return img

    # @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    with header:
        st.title("Energy Disaggregation Analysis")
        # image_file = st.file_uploader("Upload Building Image",type=["png","jpg","jpeg"])
        # if image_file is not None:
        #     file_details = {"filename":image_file.name}
        #     st.write(file_details)
        #     st.image(load_image(image_file))
        #     st.write(type(image_file))

    with dataset:
        st.header('Total Building Energy Data')
        data_Energy = st.file_uploader("Upload Total Energy Data in CSV format", type=["csv"])
        if data_Energy is not None:
            st.write(type(data_Energy))
            file_details = {"filename":data_Energy.name,"filetype":data_Energy.type,"filesize":data_Energy.size}
            st.write(file_details)
            df_energy = pd.read_csv(data_Energy,index_col=0)
            # df_energy.index= pd.to_datetime(df_energy.index)
            st.write(file_details)
            st.dataframe(df_energy)
            # st.dataframe(df_disagg)
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
        # st.subheader('Lighting and plugLoads (MJ)')
        # st.line_chart(df_seasonal_elec)

    ########### Cooling #############################################    
        bkp1 = '2018-04-28 07:00:00'
        bkp2 = '2018-04-28 08:00:00'
        bkp3 = '2018-09-24 07:00:00'
        bkp4 = '2018-09-24 08:00:00'
        # df_clg = (stl_result.seasonal.iloc[bkp2:bkp3]*stl_result.trend.iloc[bkp2:bkp3]*stl_result.resid.iloc[bkp2:bkp3])
        # df_clg = df_clg.iloc[bkp2:bkp3] - df_seasonal_elec.iloc[bkp2:bkp3]
        df_clg = (stl_result.seasonal.loc[3000:6500]*stl_result.trend.loc[3000:6500]*stl_result.resid.loc[3000:6500])
        df_clg = df_clg.loc[3000:6500] - df_seasonal_elec.loc[3000:6500]
        df_clg[df_clg < 0] = 0
        # st.subheader('Cooling (MJ)')
        # st.line_chart(df_clg)

    ########### Heating #############################################
        # df_htg1 = (stl_result.seasonal.iloc[:bkp1]*stl_result.trend.iloc[:bkp1]*stl_result.resid.iloc[:bkp1])
        # df_htg2 = (stl_result.seasonal.iloc[bkp4:]*stl_result.trend.iloc[bkp4:]*stl_result.resid.iloc[bkp4:])
        df_htg1 = (stl_result.seasonal.loc[:2999]*stl_result.trend.loc[:2999]*stl_result.resid.loc[:2999])
        df_htg2 = (stl_result.seasonal.loc[6501:]*stl_result.trend.loc[6501:]*stl_result.resid.loc[6501:])
        df_htg1 = df_htg1.loc[:2999]- df_seasonal_elec.loc[:2999]
        # df_htg1 = df_htg1.iloc[:bkp1]- df_seasonal_elec.iloc[:bkp1]
        df_htg1[df_htg1 < 0 ] = 0
        df_htg2 = df_htg2.loc[6501:]- df_seasonal_elec.loc[6501:]
        # df_htg2 = df_htg2.iloc[bkp4:]- df_seasonal_elec.iloc[bkp4:]
        df_htg2[df_htg2 < 0 ] = 0
        # st.subheader('Heating  (MJ)')
        # st.line_chart(df_htg1)
        # st.line_chart(df_htg2)
    ################################################################
        df_htg_clg = pd.concat([df_htg1, df_clg, df_htg2])
        df_disagg = pd.DataFrame()
        df_disagg['lightingPlugLoads'] = df_seasonal_elec
        # df_disagg.index = pd.to_datetime(df_disagg.index)
        df_disagg['Cooling'] = df_clg
        df_disagg['Cooling'] = df_disagg['Cooling'].fillna(0)
        df_disagg['Heating'] = df_htg_clg.values - df_disagg['Cooling'].values
        df_disagg['Heating'] = df_disagg['Heating'].fillna(0)
        # df_disagg.index = pd.to_datetime(df_disagg.index)
        # df_disagg['Hour'] = df_disagg.index.dt.hour
        csv = convert_df(df_disagg)

        st.download_button(
        "Press to Download Disaggregation Result",
        csv,
        "disagg_result.csv",
        "text/csv",
        key='download-csv'
        )
        print(df_disagg.index)
        d1 = {'Time':pd.date_range(start ='01-01-2018',end ='31-12-2018 23:00:00', freq ='H')}
        df = pd.DataFrame(d1)
        df_new = pd.DataFrame()
        df_new['Time'] = df['Time']
        df_new['ELECTRICITY(MJ)'] = df_disagg['lightingPlugLoads'].values
        df_new['Cooling(MJ)'] = df_disagg['Cooling'].values
        df_new['Heating(MJ)'] = df_disagg['Heating'].values
        df_new['Hour'] = df_new['Time'].dt.hour
        df_new['Week'] = df_new['Time'].dt.isocalendar().week
        df_new['Day Name'] = df_new['Time'].dt.day_name()
        df_new['Month'] = df_new['Time'].dt.month
        df_new['Day of Month'] = df_new['Time'].dt.day
        st.subheader('Cooling')
        st.line_chart(df_new['Cooling(MJ)'])
        st.subheader('Electricity')
        st.line_chart(df_new['ELECTRICITY(MJ)'])
        st.subheader('Heating')
        st.line_chart(df_new['Heating(MJ)'])
        print(df_new)

        ###############################################################################
        st.subheader('Lighting and Plug load Daily, Monthly and Weekly Energy Use Pattern')
        day_group = df_new.groupby(['Day Name', 'Hour']).mean().reset_index()
        days = df_new['Day Name'].unique()
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        NUM_COLORS = len(days)
        cm = plt.get_cmap('gist_rainbow')
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(111)
        plt.title("Decomposed",weight='bold')
        ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
        for i, y in enumerate(days):
            df = day_group[day_group['Day Name'] == y]
            plt.plot(df['Hour'], df['ELECTRICITY(MJ)'])
            plt.legend(df_new['Day Name'].unique(),bbox_to_anchor=(0, 1.1, 1, 0.2), loc="lower left",
            mode="expand", borderaxespad=0, ncol=3)
        plt.xlabel('Hour')
        plt.ylabel('Mean Electricity Use for \n Lighting and Plug Loads (MJ)')
        st.pyplot(fig)

        # box_plot_data=[value1,value2,value3,value4]
        fig2, ax = plt.subplots(figsize=(10,5))
        plt.suptitle('')
        df_new.boxplot(column=['ELECTRICITY(MJ)'], by='Hour', ax=ax)
        st.pyplot(fig2)
        fig3, ax = plt.subplots(figsize=(10,5))
        plt.suptitle('')
        df_new.boxplot(column=['ELECTRICITY(MJ)'], by='Month', ax=ax)
        st.pyplot(fig3)

    #################################################################
        st.subheader('Heating Daily, Monthly and Weekly Energy Use Pattern')
        day_group = df_new.groupby(['Day Name', 'Hour']).mean().reset_index()
        days = df_new['Day Name'].unique()
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        NUM_COLORS = len(days)
        cm = plt.get_cmap('gist_rainbow')
        fig4 = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(111)
        plt.title("Decomposed",weight='bold')
        ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
        for i, y in enumerate(days):
            df = day_group[day_group['Day Name'] == y]
            plt.plot(df['Hour'], df['Heating(MJ)'])
            plt.legend(df_new['Day Name'].unique(),bbox_to_anchor=(0, 1.1, 1, 0.2), loc="lower left",
            mode="expand", borderaxespad=0, ncol=3)
        plt.xlabel('Hour')
        plt.ylabel('Mean Heating Energy Use (MJ)')
        st.pyplot(fig4)

        fig5, ax = plt.subplots(figsize=(10,5))
        plt.suptitle('')
        df_new.boxplot(column=['Heating(MJ)'], by='Hour', ax=ax)
        st.pyplot(fig5)
        fig6, ax = plt.subplots(figsize=(10,5))
        plt.suptitle('')
        df_new.boxplot(column=['Heating(MJ)'], by='Month', ax=ax)
        st.pyplot(fig6)
        ############################################################################################
        st.subheader('Cooling Daily, Monthly and Weekly Energy Use Pattern')
        day_group = df_new.groupby(['Day Name', 'Hour']).mean().reset_index()
        days = df_new['Day Name'].unique()
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        NUM_COLORS = len(days)
        cm = plt.get_cmap('gist_rainbow')
        fig4 = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(111)
        plt.title("Decomposed",weight='bold')
        ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
        for i, y in enumerate(days):
            df = day_group[day_group['Day Name'] == y]
            plt.plot(df['Hour'], df['Cooling(MJ)'])
            plt.legend(df_new['Day Name'].unique(),bbox_to_anchor=(0, 1.1, 1, 0.2), loc="lower left",
            mode="expand", borderaxespad=0, ncol=3)
        plt.xlabel('Hour')
        plt.ylabel('Mean Cooling Energy Use (MJ)')
        st.pyplot(fig4)


        # box_plot_data=[value1,value2,value3,value4]
        fig8, ax = plt.subplots(figsize=(10,5))
        plt.suptitle('')
        df_new.boxplot(column=['Cooling(MJ)'], by='Hour', ax=ax)
        st.pyplot(fig8)
        fig9, ax = plt.subplots(figsize=(10,5))
        plt.suptitle('')
        df_new.boxplot(column=['Cooling(MJ)'], by='Month', ax=ax)
        st.pyplot(fig9)
        #######################################################################################
        plt.style.use('default')
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"

# %%
with tab3:
    header = st.container()
    dataset =st.container()
    st.title('Energy Disaggregation using BAS Trend Data')

    col1, col2, col3 = st.columns(3)

    with col1:
        # st.header("A cat")   
        #matplotlib inline
        header = st.container()
        dataset =st.container()
        # progress_bar = st.sidebar.progress(0)
        # status_text = st.sidebar.empty()

        # @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        # building_number = st.multiselect('Insert building number',[3642,3035])
        building_number = 3642 # Building
        st.write('The building number is ', building_number)


        search_term="HeatingValve"
        # search_term = st.selectbox(
        #     'Trend data?',
        #     ('HeatingValve', 'CoolingValve'))

        st.write('You selected:', search_term)

        st.header("Date Range")
        start_date = st.date_input('start date')
        end_date = st.date_input('end date')
        print(start_date)
        print(end_date)
        # search_term = 'HeatingValve' # Keyword search term
        # Import API puller from supplementary file

        ################################ Supplementary functions#####################
        def check_response(r):
            '''Checks to ensure the expected response is received

            The accepted response from the API from the API is response [200] this
            function outputs raises an error if any other response is received.
            '''
            if r.status_code == 200:
                return None
            else:
                raise ImportError('Received: [{}], Expected: [<Response [200]>]'.format(r))
        #########################################################################
        def check_response(r):
            '''Checks to ensure the expected response is received

            The accepted response from the API from the API is response [200] this
            function outputs raises an error if any other response is received.
            '''
            if r.status_code == 200:
                return None
            else:
                raise ImportError('Received: [{}], Expected: [<Response [200]>]'.format(r))
        ############################################################################## 
        def print_n_lines(json_var, n_lines=20, indent=2):
            '''Pretty prints n lines of json file.

            This is used to make the outputs more compact
            '''
            pretty_str = str(json.dumps(json_var, indent=indent))
            length = len(pretty_str.splitlines())

            print('First {} / {} lines of json file:\n'.format(n_lines, length))
            for line in pretty_str.splitlines()[:n_lines]:
                print(line)
            print('..............')       
        # Load text file
        with open('login_info_v1.txt') as f:
            login_info = json.loads(f.read())

        # Assign variables
        api_key = login_info['api_key']
        client_id = login_info['api_key'] # The client ID is the same as the API key
        client_secret = login_info['client_secret']
        print('Login info successfully downloaded')

        ######################################################################
        # Request access token using client_id, and client_secret
        url = 'https://login-global.coppertreeanalytics.com/oauth/token'

        my_header = {'content-type': 'application/x-www-form-urlencoded'}
        my_data = {
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret,
            'audience': 'organize'
        }
        r = requests.post(url, headers=my_header, data=my_data)
        check_response(r)
        access_token = r.json()['access_token']

        # Save in jWt header fomrat
        jwt_header = {'Authorization': 'Bearer ' + access_token}
        print('Access token has been obtained')


        #######################################################################
        # Example inputs
        # Jean Talon id= 3642


        #######################################################################

        # Initial API query gets sensor count
        url = 'https://kaizen.coppertreeanalytics.com/yana/mongo/objects/?' \
                'building={}&object_type=TL&min_device_index=1&page_size=1'.format(building_number)
        r = requests.get(url, headers=jwt_header)
        count = r.json()['meta']['pagination']['count']

        # Second API query gets full sensor list
        url = 'https://kaizen.coppertreeanalytics.com/yana/mongo/objects/?' \
                'building={}&object_type=TL&min_device_index=1&page_size={}'.format(building_number, count)
        r = requests.get(url, headers=jwt_header)

        # Convert to pandas dataframe
        df = pd.DataFrame.from_dict(r.json()['results'])[['_id', 'Object_Name']]

        print(df)
        #########################################################################]
        # Filter based on keyword
        df_filtered = df[df['Object_Name'].str.contains(search_term)].reset_index(drop=True)

        ########################### Inputs ######################
        # The download using the batch API puller
        trend_log_data = API_puller(
            trend_log_list=df_filtered,
            API_key=api_key,
            # date_range=['2019-01-01', '2020-01-01'],
            date_range=[start_date,end_date],
            resample=60
            )
        ##########################################################################
        # Plot data retrieved from the API
        print(trend_log_data)
        fig, ax = plt.subplots(figsize=(15,8))
        # fig = plt.figure(figsize=(15,8))
        st.line_chart(trend_log_data)
        # plt.plot(trend_log_data)
        # plt.xlim(trend_log_data.index[0],trend_log_data.index[1000])
        # plt.title('Data downloaded from API')
        # plt.ylabel('Heating Valve %')
        # plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=4, fancybox=True, shadow=True);
        # plt.show()s
        # st.pyplot(fig)
        #########################################################################
    with col2:
        st.header('Heating Energy Use (MJ)')
        search_term2='Jean Talon- Steam'
        df_energy = df[df['Object_Name'].str.contains(search_term2)].reset_index(drop=True)
        trend_log_data_energy = API_puller(
            trend_log_list=df_energy,
            API_key=api_key,
            # date_range=['2019-01-01', '2020-01-01'],
            date_range=[start_date,end_date],
            resample=60
        )

        print(trend_log_data_energy)
        st.line_chart(trend_log_data_energy)
        #########################################

        # search_term3='OutdoorTemp'
        # df_otemp = df[df['Object_Name'].str.contains(search_term3)].reset_index(drop=True)
        # trend_log_data_temp = API_puller(
        #     trend_log_list=df_otemp,
        #     API_key=api_key,
        #     # date_range=['2019-01-01', '2020-01-01'],
        #     date_range=[start_date,end_date],
        #     resample=60
        # )
        # print(trend_log_data_temp)
        # st.line_chart(trend_log_data_temp)

        ############################ Weather Data #######################

        st.header('Outdoor Air Temperature (degC)')
        # Available time formats: LST/UTC
        timeformat = 'LST'

        # check https://power.larc.nasa.gov/#resources for full list of parameters and modify as needed

        # ALLSKY_SFC_SW_DNI - direct normal irradiance (W/m2)
        # ALLSKY_SFC_SW_DIFF - diffuse horizontal irradiance (W/m2)
        # T2M - temperature 2 m above ground (degC)
        # RH2M - relative humidity 2 m above ground level (m/s)
        # WS2M - wind speed 2 m above ground level (m/s)

        # params = 'ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF,T2M,RH2M,WS2M'
        params = 'T2M'
        #Always use RE (renewable energy) for this purpose
        community = 'RE' 
        #Obtain LAT/LON from google maps
        location = {
            'latitude':'45.73906',
            'longitude':'-75.73906'
            }
        # Start/end time in format: 'YYYYMMDD'
        

        # sTime = '20180101'
        # eTime = '20190101'
        sTime = str(start_date)
        sTime = sTime.replace('-','')
        eTime = str(end_date)
        eTime = eTime.replace('-','')
        print(eTime)

        #%% API call for given lat/long
        cwd = Path.cwd()
        path = cwd.__str__()
        url = 'https://power.larc.nasa.gov/api/temporal/hourly/point?Time='+timeformat+'&parameters='+params+'&community='+community+'&longitude='+location['longitude']+'&latitude='+location['latitude']+'&start='+sTime+'&end='+eTime+'&format=JSON'
        data = requests.get(url)

        data = data.json()
        data = pd.DataFrame((data['properties']['parameter']))
        data = data.set_index(pd.to_datetime(data.index, format='%Y%m%d%H'))

        st.line_chart(data)
        print(data)
    ##############################Regression Model###########
   
        st.header('Heating Disaggregation Results')

        df_model = pd.merge(trend_log_data, data['T2M'], left_index=True, right_index=True)
        df_model = pd.merge(df_model,trend_log_data_energy, left_index=True, right_index=True)
        predictors = df_model.drop(['Jean Talon- Steam'], axis=1)
        response = df_model['Jean Talon- Steam']
        print(df_model)
        print(predictors.shape[1])
        print(predictors)
        # print(response)
        # print(data['T2M'])
        def rmse_HtgMdl(x): #Heating disagg model
            h = 0
            for i in range(predictors.shape[1]-1):
             h = x[i]*predictors.iloc[:,i] + h

            h = h + x[i+1]*(predictors.iloc[:,i+1]-x[i+2])*(predictors.iloc[:,i+1]-x[i+2]) + x[i+3]
            return np.sqrt(((response - h) ** 2).mean())

        x0 = np.zeros(predictors.shape[1]+2)
        b = (0.0,10)
        bnds = (b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b)
        solution_Htg = minimize(rmse_HtgMdl,x0,method='SLSQP',bounds=bnds)
        print(solution_Htg)
        print(predictors.columns)
        heating_ahu = pd.DataFrame()

        # for i in range(0,predictors.shape[1]):
        #     heating_ahu = pd.concat([heating_ahu,(predictors.iloc[:,i]*solution_Htg[i])], axis=1)

        # heating_perimeter = solution_Htg[-2] * (predictors.iloc[:,-1]-solution_Htg[-2])*(predictors.iloc[:,-1]-solution_Htg[-2]) + solution_Htg[-1] *np.ones(df_model['AH7_HeatingValve'].shape)
        # labels = trend_log_data.columns
        # fig, axs = plt.subplots(14,sharex=True, figsize=(15,30))
        # for i in range(0,14):
        #     fig.suptitle('Heating Disaggregation ')
        #     axs[i].plot(solution_Htg.x[i]*predictors.iloc[:,i], color='red',linewidth=1, linestyle='dashed')
        #     axs[i].set(ylabel='kW')
        #     axs[i].set_title(labels[i])
        #   # axs[i].set_xlim(2000,2400)
        #     fig.tight_layout(pad=2)
        #     fig.subplots_adjust(top=0.9)
        # st.plotly_chart(fig)

        df_Htg_ahu = pd.DataFrame(columns = trend_log_data.columns)

        for i in range(predictors.shape[1]-1):
            this_column = df_Htg_ahu.columns[i]
            df_Htg_ahu[this_column] = solution_Htg.x[i]*predictors.iloc[:,i]

        print(df_Htg_ahu)

        heating_disagg = pd.DataFrame()
        heating_perimeter_other = solution_Htg.x[-3]*(predictors.iloc[:,-1]-solution_Htg.x[-2])*(predictors.iloc[:,-1]-solution_Htg.x[-2]) + solution_Htg.x[-1]
        print(heating_perimeter_other)

        heating_disagg = pd.DataFrame()
        heating_disagg = pd.merge(df_Htg_ahu,heating_perimeter_other, left_index=True, right_index=True)
        heating_disagg['Perimeter Heaters and Others'] = heating_disagg['T2M']
        heating_disagg = heating_disagg.drop(['T2M'], axis=1)
        BLDGE_AREA =70,970
        print(heating_disagg)

        kpi_htg_ahu = df_Htg_ahu.sum()
        kpi_htg_perimeter_other = heating_perimeter_other.sum()
        print(kpi_htg_ahu)
        print(kpi_htg_perimeter_other)
        
        source_htg = pd.DataFrame()
        source_htg.index = kpi_htg_ahu.index
        source_htg['AHU Heating Energy(MJ)'] =kpi_htg_ahu.values
        st.bar_chart(source_htg)
        # st.bar_chart(kpi_htg_ahu)
        # label = ['Heating Energy','AH7_HeatingValve', 'AH8_HeatingValve', 'AH9_HeatingValve',
        #          'AH10_HeatingValve', 'AH11_HeatingValve', 'AH12_HeatingValve',
        #          'AH13_HeatingValve', 'AH1_HeatingValve', 'AH2_HeatingValve',
        #          'AH14_HeatingValve', 'AH15_HeatingValve', 'AH16_HeatingValve',
        #          'AH4_HeatingValve', 'AH5_HeatingValve']
        # source = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # target = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        # value = kpi_htg_ahu.values
        # print(label)
        # # data to dict, dict to sankey
        # link = dict(source = source, target = target, value = value)
        # node = dict(label = label, pad=50, thickness=7)
        # data = go.Sankey(link = link, node=node)
        # # plot
        # fig = go.Figure(data)
        # fig
    ############Cooling#############################
    with col3:
        st.header('Cooling Energy (MJ)')
        search_term3='Jean Talon- Chilled Water'
        df_cooling = df[df['Object_Name'].str.contains(search_term3)].reset_index(drop=True)
        trend_log_data_cooling = API_puller(
            trend_log_list=df_cooling,
            API_key=api_key,
            # date_range=['2019-01-01', '2020-01-01'],
            date_range=[start_date,end_date],
            resample=60
        )

        print(trend_log_data_cooling)
        st.line_chart(trend_log_data_cooling)
        ############Cooling valve position #############################
        st.header('Cooling Valve Position (%)')
        search_term4='CoolingValve'
        df_coolingvalve = df[df['Object_Name'].str.contains(search_term4)].reset_index(drop=True)
        trend_log_data_coolingvalve = API_puller(
            trend_log_list=df_coolingvalve,
            API_key=api_key,
            # date_range=['2019-01-01', '2020-01-01'],
            date_range=[start_date,end_date],
            resample=60
        )

        print(trend_log_data_coolingvalve)
        st.line_chart(trend_log_data_coolingvalve)
        
        ############Cooling regression Model###########################
        st.header('Cooling Disaggregation Results')

        df_model_cooling = pd.merge(trend_log_data_coolingvalve, data['T2M'], left_index=True, right_index=True)
        df_model_cooling = pd.merge(df_model_cooling,trend_log_data_cooling, left_index=True, right_index=True)
        predictors = df_model_cooling.drop(['Jean Talon- Chilled Water'], axis=1)
        response = df_model_cooling['Jean Talon- Chilled Water']
        print(df_model_cooling)
        print(predictors.shape[1])
        print(predictors)
        # print(response)
        # print(data['T2M'])
        def rmse_ClgMdl(x): #Heating disagg model
            h = 0
            for i in range(predictors.shape[1]-1):
                h = x[i]*predictors.iloc[:,i] + h

            h = h + x[i+1]*(predictors.iloc[:,i+1]-x[i+2])*(predictors.iloc[:,i+1]-x[i+2]) + x[i+3]
            return np.sqrt(((response - h) ** 2).mean())

        x0 = np.zeros(predictors.shape[1]+2)
        b = (0.0,10)
        bnds = (b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b)
        solution_Clg = minimize(rmse_ClgMdl,x0,method='SLSQP',bounds=bnds)
        print(solution_Clg)
        print(predictors.columns)
        cooling_ahu = pd.DataFrame()

        # for i in range(0,predictors.shape[1]):
        #     heating_ahu = pd.concat([heating_ahu,(predictors.iloc[:,i]*solution_Htg[i])], axis=1)

        # heating_perimeter = solution_Htg[-2] * (predictors.iloc[:,-1]-solution_Htg[-2])*(predictors.iloc[:,-1]-solution_Htg[-2]) + solution_Htg[-1] *np.ones(df_model['AH7_HeatingValve'].shape)
        # labels = trend_log_data.columns
        # fig, axs = plt.subplots(14,sharex=True, figsize=(15,30))
        # for i in range(0,14):
        #     fig.suptitle('Heating Disaggregation ')
        #     axs[i].plot(solution_Htg.x[i]*predictors.iloc[:,i], color='red',linewidth=1, linestyle='dashed')
        #     axs[i].set(ylabel='kW')
        #     axs[i].set_title(labels[i])
        #   # axs[i].set_xlim(2000,2400)
        #     fig.tight_layout(pad=2)
        #     fig.subplots_adjust(top=0.9)
        # st.plotly_chart(fig)

        df_Clg_ahu = pd.DataFrame(columns = trend_log_data_coolingvalve.columns)

        for i in range(predictors.shape[1]-1):
            this_column = df_Clg_ahu.columns[i]
            df_Clg_ahu[this_column] = solution_Clg.x[i]*predictors.iloc[:,i]

        print(df_Clg_ahu)

        cooling_disagg = pd.DataFrame()
        cooling_perimeter_other = solution_Clg.x[-3]*(predictors.iloc[:,-1]-solution_Clg.x[-2])*(predictors.iloc[:,-1]-solution_Clg.x[-2]) + solution_Clg.x[-1]
        print(cooling_perimeter_other)

        cooling_disagg = pd.DataFrame()
        cooling_disagg = pd.merge(df_Clg_ahu,cooling_perimeter_other, left_index=True, right_index=True)
        cooling_disagg['Perimeter Heaters and Others'] = cooling_disagg['T2M']
        cooling_disagg = cooling_disagg.drop(['T2M'], axis=1)
        BLDGE_AREA =70,970
        print(cooling_disagg)

        kpi_clg_ahu = df_Clg_ahu.sum()
        kpi_clg_perimeter_other = cooling_perimeter_other.sum()
        print(kpi_clg_ahu)
        print(kpi_clg_perimeter_other)
        
        source = pd.DataFrame()
        source.index = kpi_clg_ahu.index
        source['AHU Cooling Energy(MJ)'] =kpi_clg_ahu.values
        st.bar_chart(source)
        st.bar_chart(kpi_clg_ahu)


        label = ['Cooling Energy','AH7_CoolingValve', 'AH8_CoolingValve', 'AH9_CoolingValve',
                 'AH10_CoolingValve', 'AH11_CoolingValve', 'AH12_CoolingValve','AH13_CoolingValve', 'AH14_CoolingValve', 
                 'AH2_CoolingValve','AH15_CoolingValve', 'AH16_CoolingValve',
                 'Pent_Elev_CoolingValve','AH4_CoolingValve', 'AH5_CoolingValve',
                 'Heating Energy','AH7_HeatingValve', 'AH8_HeatingValve', 'AH9_HeatingValve', 'AH10_HeatingValve', 'AH11_HeatingValve'
                 ,'AH12_HeatingValve', 'AH13_HeatingValve', 'AH1_HeatingValve', 'AH2_HeatingValve', 'AH14_HeatingValve', 'AH15_HeatingValve',
                   'AH16_HeatingValve', 'AH4_HeatingValve', 'AH5_HeatingValve','Perimeter Heating Devices','Unmonitored Cooling']
        source = [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,  0]
        target = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        # value1 = np.array(kpi_clg_ahu.values)
        # value2 = np.array(kpi_htg_ahu.values)
        value = [0.000000e+00, 7.125857e+05,1.396961e+05, 8.346263e+05, 1.324155e+06, 3.682884e+05
                 ,8.682047e+05, 0.000000e+00, 0.000000e+00, 1.209310e+06,6.243443e+05, 
                 9.725472e-05, 1.348454e+05, 1.942690e-07, 6.837227e-08,3.019299e-09, 1.637352e-07,
                 0.000000e+00,7.620935e-05, 8.367065e-09, 5.811317e+03, 1.820391e-07,1.421369e+06,
                 6.857399e+05,8378372.945483186,2481835.854260325]
        print(kpi_htg_ahu)
        print(kpi_clg_ahu)
        print(kpi_clg_perimeter_other)
        print(kpi_htg_perimeter_other)
        # value = np.concatenate(value1, value2)
        print(value)
        # data to dict, dict to sankey
        link = dict(source = source, target = target, value = value)
        node = dict(label = label, pad=50, thickness=7)
        data = go.Sankey(link = link, node=node)
        # plot
        fig2 = go.Figure(data)
        fig2