import streamlit as st
import streamlit.components.v1 as components
from streamlit_echarts import st_echarts
from PIL import Image
import pandas as pd
import numpy as np
import altair as alt
import joblib
from lightgbm import LGBMClassifier as lgb
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import requests as re
import json


data_info = pd.read_csv("data_info.csv")
data_info.set_index('SK_ID_CURR', inplace= True)
# data for model   
data_api = pd.read_csv("data_api.csv")
data_api.set_index('SK_ID_CURR', inplace = True)
sav_model = joblib.load('final_model.sav')
features = data_api.columns


mylist = data_info.index.tolist()
best_threshold = 0.49


def get_info(id_client): 
    data_client = data_info[data_info.index== id_client]
    return(data_client)

def get_pred_api(id_customer) :
    query = {'id_client' :id_customer}
    repo = re.get('http://35.180.66.152'+'/prediction/', params= query)
    target= repo.json()['prediction'] 
    proba = repo.json()['proba_yes']
    return(target, proba)


def get_info_client_api(id_customer) :
    query = {'id_client' :id_customer}
    repo = re.get('http://35.180.66.152'+ '/clinet_info/', params= query)
    result = repo.json()
    return(result)

    #Plot distribution d'une variable vs Target
def plot_distribution_comp(feature,value):
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    fig, ax = plt.subplots()
    #plt.figure(figsize=(20,6))
    t0 = data_info.loc[data_info['TARGET'] == 0]
    t1 = data_info.loc[data_info['TARGET'] == 1]
    ax= sns.kdeplot(t0[feature].dropna(), color='green', bw_adjust=0.5, label="Credit paid")
    ax= sns.kdeplot(t1[feature].dropna(), color='red', bw_adjust=0.5, label="Credit not paid")
    plt.title("Distribution of %s" % feature, fontsize= 15)
    plt.axvline(value, color='b' ,linewidth = 0.8, alpha= 0.8)
    plt.legend(fontsize=10)
    plt.show()

def get_shap_explainer(id_client):    
    #shap_client = data_api[data_api.index==id_client]
    explainer = shap.TreeExplainer(sav_model)
    shap_vals = explainer.shap_values(data_api)
    expected_vals = explainer.expected_value
    return(shap_vals, expected_vals)    

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height) 
    
st.set_page_config(page_title="Project 7 Dashboard",
                   initial_sidebar_state="expanded",
                   layout="wide")
st.markdown("Creator : **_Victoire MOHEBI_**")

# Side bar
with st.sidebar:
    image_HC = Image.open("logo.jpg")
    st.image(image_HC, width=300)

# CHECKBOX 
home = st.sidebar.checkbox("Home Page")
model = st.sidebar.checkbox("Model information")
customer_info= st.sidebar.checkbox("Customers' info & Comparaison")
customer_result = st.sidebar.checkbox("Prediction result")

if model :   
    menue = ['Model', 'Feature importance', 'Shap',]
    choice = st.selectbox('Menue', menue)
    if choice == 'Model' :     
        st.markdown('<p style="font-family:Arial; font-weight: bold; color:red; font-size: 24px;">Model Information</p>', unsafe_allow_html=True)    
        st.markdown('<p style="font-family:Arial; color:black; font-size: 15px;">We have used **LGBMClassifier** as a machine learning model to classify customers</p>', unsafe_allow_html=True)    
        st.markdown('<p style="font-family:Arial; color:black; font-size: 15px;">To evaluate our model we have used **ROC-AUC**(area under the ROC curve) and **Fbeta-Score**</p>', unsafe_allow_html=True)    
        col1, col2 = st.columns(2) 
        with col1 :
            col1.markdown('<p style="font-family:Arial; color:green; font-size: 25px;">AUC_ROC : 76%</p>', unsafe_allow_html=True)    
        with col2:                
            col2.markdown('<p style="font-family:Arial; color:green; font-size: 25px;">F5_Score : 60%</p>', unsafe_allow_html=True)    
        st.write('')    
        cm_image = Image.open("confusion_matrix.jpg")
        st.image(cm_image)
        
    elif choice == 'Feature importance':
        #st.title('Feature Importance')
        st.markdown('<p style="font-family:Arial;font-weight: bold; color:#0066cc; font-size: 24px;">Top 20 of the most important features for the model</p>', unsafe_allow_html=True)    
        fi = Image.open("fi.jpg")
        st.image(fi)

    else :
        #st.title('Shaply Values') 
        st.markdown('<p style="font-family:Arial; font-weight: bold; color:green; font-size: 24px;">Top 20 features of shaply values the model</p>', unsafe_allow_html=True)    
        shap_plot = Image.open("shap_globale.jpg")
        st.image(shap_plot)

               
elif customer_info:  
    st.subheader("Select a customer ID")
    choice_id = st.selectbox('', mylist) 
    for i in mylist :
        if choice_id == i:
            customer_data = get_info(choice_id)
            customer_df = customer_data.drop('TARGET', axis= 1)
            customer_df = customer_df.to_dict('index')    
            target_i, probability_default_i = get_pred_api(choice_id) 
            style_title = '<p style="font-family:Arial; font-weight: bold;color:Black; font-size: 30px;">Customer basic information</p>'
            st.markdown(style_title,unsafe_allow_html=True)
            #text =  "Customer's basic information"
            #st.markdown(f'<p style="color:#0066cc;">{text}</p>', unsafe_allow_html=True)
            st.write(customer_df)
            if target_i == '1.0' :
                score_value1= str(probability_default_i)
                style1 = '<p style="font-family:Arial; color:red;font-size: 25px;">Credit refused</p>'
                st.markdown(style1, unsafe_allow_html=True)   
                score_style1= '<p style="font-family:Arial; color:black; font-size: 25px;">The score of non_repayment of the loan for the customer is :</p>'
                st.markdown("The score of no_repayment of the loan for the customer is : " + score_value1, unsafe_allow_html=True)      
                #st.metric('',value = probability_default_i)
                #st.success("CREDIT REFUSED ! ", icon= '❌')                    
                         
            else :
                score_value0 = str(probability_default_i)
                st.markdown(f'<p style="font-family:Arial;color:green;font-size:25px;">Credit accepted</p>', unsafe_allow_html=True) 
               # score_style0= '<p style="font-family:Arial; color:black; font-size:10px;">The score of non_repayment of the loan for the customer is :</p>'
                st.markdown("The score of no_repayment of the loan for the customer is : " + score_value0,unsafe_allow_html=True)     
               # st.metric('',value = probability_default_i)
                #st.success("CREDIT ACCEPTED ! ", icon="✅")  
                         
            feauture_select = ['AMT_ANNUITY','AMT_GOODS_PRICE','ANNUITY_INCOME%','PAYMENT_RATE',
                                    'AGE_CLIENT','YEARS_EMPLOYED', 'CNT_FAM_MEMBERS']  
            f_style= '<p style="font-family:Arial; font-weight: bold;color:Black; font-size: 30px;">Select the feature</p>'
            st.markdown(f_style, unsafe_allow_html=True)
            #st.subheader("Which feature do you want to show?")
            feat_to_show = st.selectbox('', options= feauture_select)
            for var in feauture_select :
                if feat_to_show == var :  
                    val = int(customer_data[var].values)               
                    fig = plot_distribution_comp(var,val)  
                    st.set_option('deprecation.showPyplotGlobalUse', False)        
                    st.pyplot(fig)               
elif customer_result : 
    st.title("Select an ID")
    choice_id = st.selectbox('Choose a customer ID', options = mylist, index=0)  
    for i in mylist:   
        if choice_id == i: 
            client_info = get_info(choice_id)
            target, probability_default = get_pred_api(choice_id)       
            if target == '1.0':
                res_title1 = '<p style="font-family:Arial; font-weight: bold;color:Red; font-size: 30px;">Credit Refused</p>'
                st.markdown(res_title1,unsafe_allow_html=True)
               # st.markdown(f'<p style="color:#FF0000;font-size:24px;">{res_t}</p>', unsafe_allow_html=True) 
                fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = 100*probability_default,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probability of no_repayment of the loan%", 'font': {'size': 24}},       
                gauge = {
                'axis': {'range': [None, 100]},
                    'bar': {'color': "red"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray"              
                    }))
                st.plotly_chart(fig, use_container_width=True)
                shap_values0, expected_values0 = get_shap_explainer(i) 
                shap_title0 = '<p style="font-family:Arial; font-weight: bold;color:Black; font-size: 30px;">Assessing feature importance based on Shap values</p>'
               # shap_title1 = '<p style="font-family:Arial; font-weight: bold;color:Black; font-size: 20px;">Shap Force Plot</p>'
                shap_title2 = '<p style="font-family:Arial; font-weight: bold;color:Black; font-size: 20px;">Shap bar plot</p>'         
                st.markdown(shap_title0,unsafe_allow_html=True)
                #st.markdown(shap_title1,unsafe_allow_html=True)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                #plt.clf()
               # st_shap(shap.force_plot(base_value= expected_values0[1], shap_values =shap_values0[1][choice_id - 100002],link= 'logit', features=list(features), figsize =(20,3),plot_cmap="PkYg"))
                st.markdown(shap_title2,unsafe_allow_html=True)
                #st.markdown(shap_title2,unsafe_allow_html=True)
                shap.bar_plot(shap_values0[1][choice_id - 100002], feature_names=list(features), max_display=10)
                st.pyplot(bbox_inches='tight')
                #st.set_option('deprecation.showPyplotGlobalUse', False) 
                #st.plotly_chart(shap_force_plot1,use_container_width=True)                                                        
            else :                                                 
                res_title2 = '<p style="font-family:Arial; font-weight: bold;color:Green; font-size: 30px;">Credit Accepted</p>'
                st.markdown(res_title2,unsafe_allow_html=True)
                fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = 100*probability_default,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probability of no_repayment of the loan%", 'font': {'size': 24}},       
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "green"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray"              
                    }))
                st.plotly_chart(fig, use_container_width=True)
                shap_values0, expected_values0 = get_shap_explainer(i) 
                shap_title0 = '<p style="font-family:Arial; font-weight: bold;color:Black; font-size: 30px;">Assessing feature importance based on Shap values</p>'
                #shap_title1 = '<p style="font-family:Arial; font-weight: bold;color:Black; font-size: 20px;">Shap Force Plot</p>'
                shap_title2 = '<p style="font-family:Arial; font-weight: bold;color:Black; font-size: 20px;">Shap bar plot</p>'         
                st.markdown(shap_title0,unsafe_allow_html=True)
                #st.markdown(shap_title1,unsafe_allow_html=True)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                #plt.clf()
                #st_shap(shap.force_plot(base_value= expected_values0[1], shap_values =shap_values0[1][choice_id - 100002],link= 'logit', features=list(features), figsize =(20,3),plot_cmap="PkYg"))
                st.markdown(shap_title2,unsafe_allow_html=True)
                #st.markdown(shap_title2,unsafe_allow_html=True)
                shap.bar_plot(shap_values0[1][choice_id - 100002], feature_names=list(features), max_display=10)
                st.pyplot(bbox_inches='tight')
                                            
else: 
    st.header('Welcome to Home Credit Default Risk Prediction')
    image = Image.open("home credit.jpg")
    col1, col2, col3 = st.columns([1,10,1])    
    with col1:
        st.write("")
    with col2:
            st.image(image, width=600)
    with col3:
        st.write("")
