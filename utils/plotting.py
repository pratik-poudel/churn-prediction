import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from plotly.offline import iplot
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


@st.cache()
def plot_distribution(dataframe, var_select=None, bins=1.0): 
    df = dataframe.copy()
    # Calculate the correlation coefficient between the new variable and the target
    df["Churn"] = df['Churn'].replace({"Yes":1, "No":0})
    tmp_churn = df[df['Churn'] == 1]
    tmp_no_churn = df[df['Churn'] == 0]    
    corr = df['Churn'].corr(df[var_select])
    corr = np.round(corr,3)
    tmp1 = tmp_churn[var_select].dropna()
    tmp2 = tmp_no_churn[var_select].dropna()
    hist_data = [tmp1, tmp2]
    
    group_labels = ['Churned', 'Not churned']
    colors = ['indianred','seagreen' ]

    fig = ff.create_distplot(hist_data,
                             group_labels,
                             colors = colors, 
                             show_hist = True,
                             curve_type='kde', 
                             bin_size = bins
                            )
    
    fig['layout'].update(title = var_select+' '+'(Correlation with Churn ='+ str(corr)+')')
    # iplot(fig, filename = 'Density plot')

    return fig


@st.cache()
def binary_ploting_distributions(dataframe, cat_col):
    df = dataframe.copy()
    
    fig = make_subplots(rows=1,cols=2,print_grid=True,horizontal_spacing=0.2, 
    subplot_titles=("Distribution and % Churn", 
                                              f'Mean Monthly Charges of {cat_col}'))

    df["Churn"] = df['Churn'].replace({"Yes":1, "No":0})
    tmp_churn = df[df['Churn'] == 1]
    tmp_no_churn = df[df['Churn'] == 0]
    # calculate churn / total count of categorical variables 
    tmp_attr = round(tmp_churn[cat_col].value_counts().sort_index() / df[cat_col].value_counts().sort_index(),2)*100

    t1 = tmp_churn[cat_col].value_counts().sort_index()
    t2 = df[cat_col].value_counts().sort_index().rename(f'Total-{cat_col}')
    data_points = pd.concat([t1, t2], axis=1)

    data_points.columns = ['Churn Count', 'Total']
    data_points['Churned %'] =  round((data_points['Churn Count'] / data_points['Total']) * 100, 2)
    


    trace1 = go.Bar(
        x=tmp_churn[cat_col].value_counts().sort_index().index,
        y=tmp_churn[cat_col].value_counts().sort_index().values,
        name='Churned',opacity = 0.8, marker=dict(
            color='indianred',
            line=dict(color='#000000',width=1)))

    trace2 = go.Bar(
        x=tmp_no_churn[cat_col].value_counts().sort_index().index,
        y=tmp_no_churn[cat_col].value_counts().sort_index().values,
        name='Not Churned', opacity = 0.8, 
        marker=dict(
            color='seagreen',
            line=dict(color='#000000',
                      width=1)
        )
    )

    trace3 =  go.Scatter(   
        x=tmp_attr.sort_index().index,
        y=tmp_attr.sort_index().values,
        yaxis = 'y2',
        name='% Churn', opacity = 0.6, 
        marker=dict(
            color='black',
            line=dict(color='#000000',
                      width=2 )
        )
    )
    df_tmp = (df.groupby(['Churn', cat_col])['MonthlyCharges'].mean().reset_index())
    tmp_churn = df_tmp[df_tmp['Churn'] == 1]
    tmp_no_churn = df_tmp[df_tmp['Churn'] == 0]

    df_tmp = (df.groupby(['Churn', cat_col])['MonthlyCharges'].mean()).unstack('Churn').reset_index()
    df_tmp['diff_rate'] = round((df_tmp[1] / df_tmp[0]) - 1,2) * 100

    trace4 = go.Bar(
        x=tmp_churn[cat_col],
        y=tmp_churn['MonthlyCharges'], showlegend=False,
        name='Mean Charge Churn',opacity = 0.8, marker=dict(
            color='indianred',
            line=dict(color='#000000',width=1)))

    trace5 = go.Bar(
        x=tmp_no_churn[cat_col],
        y=tmp_no_churn['MonthlyCharges'],showlegend=False,
        name='Mean Charge NoChurn', opacity = 0.8, 
        marker=dict(
            color='seagreen',
            line=dict(color='#000000',
                      width=1)
        )
    )

    trace6 =  go.Scatter(   
        x=df_tmp[cat_col],
        y=df_tmp['diff_rate'],
        yaxis = 'y2',
        name='% Diff Churn', opacity = 0.6, 
        marker=dict(
            color='black',
            line=dict(color='#000000',
                      width=5 )
        )
    )

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 1) 
    fig.append_trace(trace3, 1, 1)
    fig.append_trace(trace4, 1, 2)
    fig.append_trace(trace5, 1, 2)
    fig.append_trace(trace6, 1, 2) 

    fig['data'][2].update(yaxis='y3')
    fig['data'][5].update(yaxis='y4')

    fig['layout']['xaxis'].update(autorange=True,
                                   tickfont=dict(size= 10), 
                                   title= f'{cat_col}', 
                                   type= 'category',
                                  )
    fig['layout']['yaxis'].update(title= 'Count')

    fig['layout']['xaxis2'].update(autorange=True,
                                   tickfont=dict(size= 10), 
                                   title= f'{cat_col}', 
                                   type= 'category',
                                  )
    fig['layout']['yaxis2'].update( title= 'Mean Monthly Charges' )

    fig['layout']['yaxis3']=dict(range= [0, 100], #right y-axis in subplot (1,1)
                              overlaying= 'y', 
                              anchor= 'x', 
                              side= 'right', 
                              showgrid= False, 
                              title= '%Churn Ratio'
                             )

    #Insert a new key, yaxis4, and the associated value:
    fig['layout']['yaxis4']=dict(range= [-20, 100], #right y-axis in the subplot (1,2)
                              overlaying= 'y2', 
                              anchor= 'x2', 
                              side= 'right', 
                              showgrid= False, 
                              title= 'Monthly % Difference'
                             )
    fig['layout']['title'] = f" Distributions  of {cat_col} (Total Churned / Not Churned and % of Total Churned / Not Churned)"
    fig['layout']['height'] = 500
    fig['layout']['width'] = 1200
    
    df_tmp.columns = [cat_col, 'NotChurned', "Churned", "Difference"]
    return fig, data_points, df_tmp



color_op = ['#5527A0', '#BB93D7', '#834CF7', '#6C941E', '#93EAEA', '#7425FF', '#F2098A', '#7E87AC', 
            '#EBE36F', '#7FD394', '#49C35D', '#3058EE', '#44FDCF', '#A38F85', '#C4CEE0', '#B63A05', 
            '#4856BF', '#F0DB1B', '#9FDBD9', '#B123AC']


def plot_pie(dataframe, df_cat, df_value, limit=15):

    df = dataframe.copy()

    df["Churn"] = df['Churn'].replace({"Yes":1, "No":0})
    tmp_churn = df[df['Churn'] == 1].groupby(df_cat)[df_value].sum().nlargest(limit).to_frame().reset_index()
    tmp_no_churn = df[df['Churn'] == 0].groupby(df_cat)[df_value].sum().nlargest(limit).to_frame().reset_index()

    p1= go.Pie(labels = tmp_churn[df_cat], values=tmp_churn[df_value], name='Churn', hole=0.5, domain= {'x': [0, .5]})
    p2 = go.Pie(labels = tmp_no_churn[df_cat], values=tmp_no_churn[df_value], name='No Churn', hole=0.5,domain= {'x': [.5, 1]} )
    layout = dict(title= f"Total {df_value} by {df_cat}" , height=450,width=1200, font=dict(size=15),
                    annotations = [
                        dict(
                            x=.22, y=.5,
                            text='Churn', 
                            showarrow=False,
                            font=dict(size=20)
                        ),

                        dict(
                            x=.8, y=.5,
                            text='No Churn', 
                            showarrow=False,
                            font=dict(size=20)
                        ),
                        
            ])

    fig = dict(data=[p1, p2], layout=layout)


        
    return fig, tmp_churn, tmp_no_churn 