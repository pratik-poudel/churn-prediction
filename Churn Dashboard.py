import streamlit as st

st.set_page_config(page_title="Customer Churn Dashboard",layout="wide")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.express as px
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from collections import Counter
from scipy.stats import chi2_contingency
from utils.plotting import plot_distribution, binary_ploting_distributions, plot_pie

@st.cache(suppress_st_warning=True)
def load_data(path):
    dataframe =pd.read_csv(path)
    df = dataframe.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.columns = ['customerID', 'Gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'Tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']
    df = df.drop(columns = ['customerID'])
    cat_cols = df.select_dtypes('object').columns.to_list()
    for col in cat_cols:
        df[col] = df[col].str.title()
    df = df.dropna()
    return df


data  = load_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

def monthly_revenue_lost(df):
    df = df.copy()
    churn_df = df[df['Churn'] == "Yes"]
    no_churn_df = df[df['Churn'] == "No"]
    retained_revenue = no_churn_df['MonthlyCharges'].sum()
    not_retained = churn_df['MonthlyCharges'].sum()
    total_revenue = df['MonthlyCharges'].sum()

    pct_retained = round((retained_revenue / total_revenue) * 100, 2)
    pct_not_retained = round((not_retained / total_revenue) * 100, 2)

    return int(total_revenue), pct_retained, pct_not_retained

####################### Metrics

st.title("Customer Churn Dashboard")


churn_pct = round(data['Churn'].value_counts(normalize=True) * 100,2)

total_customers, churn_col, pct_churned, total_revenue_col, net_retention_rate_col, revenue_lost_col = st.columns(6)

total_customers.metric("Total Customers",  len(data))
churn_col.metric("Customers Churned", data['Churn'].value_counts()[1] )
pct_churned.metric("% Customers Churned ", str(churn_pct[1]))
total_revenue, rev_retained, rev_not_retained = monthly_revenue_lost(data)
total_revenue_col.metric("Total Revenue ", total_revenue)
net_retention_rate_col.metric(" % Revenue Retained", rev_retained)
revenue_lost_col.metric(" % Revenue Lost",rev_not_retained )

st.markdown("---")


## Monthly Charges & Total Charges Distribution

mnth_chages_dist, total_charges_dist  = st.columns(2)

with st.container():
    mnth_fig = plot_distribution(data, var_select='MonthlyCharges', bins=5)
    total_fig = plot_distribution(data, var_select='TotalCharges', bins=50)
    mnth_chages_dist.plotly_chart(mnth_fig, use_container_width=True)
    total_charges_dist.plotly_chart(total_fig, use_container_width=True)

st.markdown("---")

categorical_columns = data.select_dtypes('object').columns[:-1]

c1, c2 = st.columns([2, 1])
with c1:
    option = st.selectbox('Select Columns', categorical_columns)


binary_fig_cont, cat_col_churn_pct, mean_monthly_charges_cat = binary_ploting_distributions(data, option)


#Plot distribution
st.plotly_chart(binary_fig_cont)

data1, data2 = st.columns(2)
with data1.expander(f'Show Data for {option} Distribution'):
    st.dataframe(cat_col_churn_pct.style.format("{:.0f}"))
with data2.expander(f"Show Data for {option} Mean Monthly Charges"):
    st.dataframe(mean_monthly_charges_cat.style.format({'NotChurned': '{:.1f}', 'Churned': '{:.2f}', 'Difference': '{:.1f}'}))


# df_cat = 'InternetService'
# df_value = "MonthlyCharges"
# limit = 15

# df = data.copy()
# df["Churn"] = df['Churn'].replace({"Yes":1, "No":0})
# tmp_churn = df[df['Churn'] == 1].groupby(df_cat)[df_value].sum().nlargest(limit).to_frame().reset_index()
# tmp_no_churn = df[df['Churn'] == 0].groupby(df_cat)[df_value].sum().nlargest(limit).to_frame().reset_index()

# st.header("Churn")
# st.dataframe(tmp_churn)

# st.header("No Churn")
# st.dataframe(tmp_no_churn)




#Plot Pie charts 

st.markdown("---")
st.header("Churner Profile vs Non Churner Profile")

c1, c2 = st.columns([2, 1])

with c1:
    option = st.selectbox('Select Columns', categorical_columns, key='pie')


# pie_fig = plot_pie(data, "InternetService", 'MonthlyCharges', "Internet Services Monthly Charges by Churn", limit=10)
# st.plotly_chart(pie_fig )

# pie_fig = plot_pie(data, "DeviceProtection", 'MonthlyCharges', "Type of Contract by Churn or not with Ratio of Monthly Charges", limit=10)
# st.plotly_chart(pie_fig)



df_value = 'MonthlyCharges'
fig, tmp_churn, tmp_no_churn = plot_pie(data, option, df_value )
st.plotly_chart(fig)

data1, data2 = st.columns(2)
with data1.expander(f'Churner"s Data {df_value} for {option} '):
    st.dataframe(tmp_churn)
    # .style.format("{:.0f}"))
with data2.expander(f"Non Churner's Data {df_value} for {option}"):
    st.dataframe(tmp_no_churn)
    # .style.format({'NotChurned': '{:.1f}', 'Churned': '{:.2f}', 'Difference': '{:.1f}'}))
