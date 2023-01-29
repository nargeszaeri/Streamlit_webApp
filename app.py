from matplotlib.pyplot import title
# import seaborn as sns
import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


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

@st.experimental_memo
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
    st.subheader('Lighting and plugLoads (MJ)')
    st.line_chart(df_seasonal_elec)

########### Cooling #############################################    
    bkp1 = '2018-04-28 07:00:00'
    bkp2 = '2018-04-28 08:00:00'
    bkp3 = '2018-09-24 07:00:00'
    bkp4 = '2018-09-24 08:00:00'
    df_clg = (stl_result.seasonal.loc[bkp2:bkp3]*stl_result.trend.loc[bkp2:bkp3]*stl_result.resid[bkp2:bkp3])
    df_clg = df_clg.loc[bkp2:bkp3] - df_seasonal_elec.loc[bkp2:bkp3]
    df_clg[df_clg < 0] = 0
    st.subheader('Cooling (MJ)')
    st.line_chart(df_clg)

########### Heating #############################################
    df_htg1 = (stl_result.seasonal.loc[:bkp1]*stl_result.trend.loc[:bkp1]*stl_result.resid.loc[:bkp1])
    df_htg2 = (stl_result.seasonal.loc[bkp4:]*stl_result.trend.loc[bkp4:]*stl_result.resid.loc[bkp4:])
    df_htg1 = df_htg1.loc[:bkp1]- df_seasonal_elec.loc[:bkp1]
    df_htg1[df_htg1 < 0 ] = 0
    df_htg2 = df_htg2.loc[bkp4:]- df_seasonal_elec.loc[bkp4:]
    df_htg2[df_htg2 < 0 ] = 0
    st.subheader('Heating  (MJ)')
    st.line_chart(df_htg1)
    st.line_chart(df_htg2)
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


    # day_group = df_new.groupby(['Day Name', 'Hour']).mean().reset_index()
    # days = df_new['Day Name'].unique()
    # NUM_COLORS = len(days)
    # # cm = plt.get_cmap('gist_rainbow')
    # cm = plt.get_cmap('tab10')

    # fig11, axes = plt.subplots(3,1, figsize=(12,6))
    # # ax = fig.add_subplot(121)
    # plt.subplot(311)
    # plt.title("Decomposed",weight='bold')
    # ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    # for i, y in enumerate(days):
    #     df = day_group[day_group['Day Name'] == y]
    #     plt.plot(df['Hour'], df['Cooling(MJ)'])
    #     plt.ylabel('Mean Lighting \n & Plug Load Energy (MJ)')
    #     plt.legend(df_new['Day Name'].unique(),bbox_to_anchor=(0, 1.3, 1, 0.2), loc="lower left",
    #     mode="expand", borderaxespad=0, ncol=3)
    # plt.subplot(312)
    # plt.title("Submetered",weight='bold')
    # ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    # for i, y in enumerate(days):
    #     df = day_group[day_group['Day Name'] == y]
    #     plt.plot(df['Hour'], df['ELECTRICITY(MJ)'],linestyle='dashed') 
    #     # plt.xlabel('Hour')
    # plt.ylabel('Mean Lighting \n & Plug Load Energy (MJ)')
    # plt.legend(df_new['Day Name'].unique(),bbox_to_anchor=(0, 1.3, 1, 0.2), loc="lower left",
    # mode="expand", borderaxespad=0, ncol=3)
    # plt.subplot(313)
    # ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    # for i, y in enumerate(days):
    #     df = day_group[day_group['Day Name'] == y]
    #     plt.plot(df['Hour'], df['Heating(MJ)']) 
    # plt.ylabel('Mean Heating \n Energy (MJ)')
    # plt.xlabel('Hour')
    # plt.ylabel('Mean Cooling \n Energy (MJ)')
    # st.pyplot(fig11)
