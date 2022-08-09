import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from feature_engine.encoding import OneHotEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from traitlets import default



add_selectbox = st.sidebar.selectbox(
	"How would you like to predict?", ("Online", "Batch"))
st.sidebar.info('This app is created to predict Customer Churn')



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
    df.SeniorCitizen.replace({0: "No", 1: "Yes"}, inplace=True)
    cat_cols = df.select_dtypes('object').columns.to_list()
    for col in cat_cols:
        df[col] = df[col].str.title()
    df = df.dropna().reset_index(drop=True)
    return df



data  = load_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
st.write("Loaded data Shape", data.shape)

#### Utility Functions 


def create_sub_df(dataframe, colname, col_type_name):
    tmp = dataframe[dataframe[colname] == col_type_name]
    # tmp = tmp.drop(columns=[colname, 'TotalCharges', 'Tenure']).reset_index(drop=True)
    # tmp.SeniorCitizen.replace({0: "No", 1: "Yes"}, inplace=True)
    return tmp

# def make_X_Y(dataframe, target):
#     X = dataframe.drop(columns=[target])
#     y = dataframe[target]
#     return X, y

def make_cat_num_cols(dataframe ):
    cat_cols = dataframe.select_dtypes('object').columns 
    num_cols = [c for c in dataframe.columns if not c in cat_cols]
    return cat_cols, num_cols

def transform_cat_cols(dataframe, cols_list, encoder=None):
    cat_data = encoder.fit_transform(dataframe[cols_list])
    return cat_data

def transform_num_cols(dataframe, cols_list, encoder=None):
    num_data = encoder.fit_transform(dataframe[cols_list])
    num_data = pd.DataFrame(num_data, columns=cols_list)
    return num_data



# Categorical Encoding and Scaling Numerical Columns
ohe = OneHotEncoder(drop_last=True)
scaler = StandardScaler()


## Prepare Streamlit App

c1,c2, c3  = st.columns([1, 1, 5])


all_cols_select = [col for col in data.columns if not 'Churn' in col] + ["None"]

filter_by_col = c1.selectbox("Select Columns to filter", all_cols_select, index=len(all_cols_select)-1)


if filter_by_col == "None":
    sub_df = data.copy()
    st.write("SUB DF Copied ", sub_df.shape)
    filter_by_col_unique = c2.selectbox('Select Unique Values' ,['None'])
    include_cols = c3.multiselect("Please Specify Which Columns to include", 
    options = [f for f in all_cols_select if not filter_by_col in f],
    default = [f for f in all_cols_select if not filter_by_col in f])
    

    X = sub_df.drop(columns=['Churn'])[include_cols]
    st.write("X Shape ", X.shape)
    with st.expander("Show Glimpse of Data Selected"):
        st.subheader(f"Total data points {X.shape[0]}")
        st.dataframe(X.head())
    y = sub_df['Churn']

else:
    filter_by_col_unique = c2.selectbox('Select Unique Values' , data[filter_by_col].unique())
    include_cols = c3.multiselect("Please Specify Which Columns to include", 
    options = [f for f in all_cols_select if not filter_by_col in f and not "None" in f],
    default = [f for f in all_cols_select if not filter_by_col in f and not "None" in f])

    # Create Sub dataframe based on user inputted filters
    sub_df = create_sub_df(data, filter_by_col, filter_by_col_unique)
    sub_df = sub_df[include_cols]
    with st.expander("Show Data Selected"):
        st.subheader(f"Total data points {sub_df.shape[0]}")
    # st.subheader("Glimpse Of Data")
    # st.dataframe(data.loc[sub_df.index].head(10))
        st.dataframe(sub_df.head())


    target_df = data.loc[sub_df.index]['Churn']
    # st.dataframe(target_df)

    assert(sub_df.shape[0] == target_df.shape[0])
    X, y = sub_df.reset_index(drop=True), target_df.reset_index(drop=True)



# st.write(X)
cat_cols, num_cols = make_cat_num_cols(X)
# st.write(cat_cols)
# st.write(num_cols)

cat_data = transform_cat_cols(X, cat_cols, encoder=ohe)
# st.dataframe(cat_data)


if len(num_cols) >=1:

    num_data= transform_num_cols(X, num_cols, encoder=scaler)
    # st.write("Num data", num_data.shape)
    whole_data = pd.concat([cat_data, num_data], axis=1)
    # st.write("Whole data", whole_data.shape)
else:
    whole_data = pd.concat([cat_data], axis=1)

# st.write(len(num_cols))
# st.write(X.shape)
# st.write(whole_data.shape)

# st.dataframe(whole_data)

# month_df = create_sub_df(df, 'Contract', 'Month-To-Month')
# X, y = make_X_Y(month_df, 'Churn')


X_train, X_val, y_train, y_val = train_test_split(whole_data, y, test_size=0.2, random_state=2024,stratify=y)


def train_model(X_train,X_val, y_train, y_val , model):

    model.fit(X_train, y_train)
    preds= model.predict(X_val)
    preds_proba = model.predict_proba(X_val)[:,1]

    return preds, preds_proba


log_reg = LogisticRegression(class_weight='balanced', max_iter=500, random_state=2024)

preds, preds_proba = train_model(X_train,X_val, y_train, y_val , model=log_reg)

st.markdown("---")
st.subheader("Plot Showing importance of Features")
imp_df = pd.DataFrame(log_reg.coef_[0], index=X_train.columns, columns=['Score'])
imp_df.sort_values(by='Score',ascending=False, inplace=True)
fig = px.bar(imp_df.head(20), x = 'Score', height=500)
fig.update_layout(yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig, use_container_width=True)
# fig.show()

st.dataframe(imp_df)