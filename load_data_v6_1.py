from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
        import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from Disparate_impact_streamlit_demo import *

st.set_page_config(layout='wide')

def plot_bias(dict,name1,name2,column1,column2,val1,val2):
    bars = []
    bars.append(go.Bar(x=[dict[column1][val1]],
                    y=[column1],
                    marker={'color': 'rgb(250,145,90)'},
                    name=name1,
                    orientation = 'h',
                    width=[0.4,0.4,0.4],
                    textposition='auto'
                    ))

    bars.append(go.Bar(x=[dict[column2][val2]],
                    y=[column2],
                    marker={'color': 'rgb(84, 194, 232)'},
                    name=name2,
                    orientation = 'h',
                    width=[0.4,0.4,0.4],
                    textposition='auto'))

    config = {'displayModeBar': False}

    fig = go.FigureWidget(data=bars)
    fig.layout.height = 350
    fig.layout.width = 350
    return fig

def select_mitigation_algorithm(dataset_dict):
    st.markdown("A variety of algorithms can be used to mitigate bias.\
                The choice of which to use depends on whether you want \
                to fix the data (pre-process), the classifier (in-process), \
                or the predictions (post-process).")
    st.markdown("** Pre-Processing mitigation algorithms**")
    algo_name = st.radio("Select the mitigation Algorithm",('Disperate Impact Remover', 'Reweighing'))
    if algo_name == 'Disperate Impact Remover':
        st.markdown("The algorithm corrects values for \
            imbalanced selection rates \
            between unprivileged and privileged groups \
             at various levels of repair.")
    else :
        st.markdown("The algorithm uses classifier agnostic iterative approach \
            i.e.first fully train a classifier  \
            based on uniform weights  \
             and then appropriately readjust.")
    dataset_dict['algo_name'] = algo_name
    return dataset_dict

        
    
def bias_check():
    processed_data = pd.read_csv("C:\\Users\\Joshika\\Desktop\\Navya\\CBA\\Bias Mitigation\\demo9\\AIF_final_adult_data_processed.csv")
    processed_data['sex'] = processed_data['sex'].astype(str) 
    processed_data['race'] = processed_data['race'].astype(str)
    columns = processed_data.select_dtypes(include=['category', object]).columns
    st.markdown("**Unbiased** is a condition where the data doesn't show any discrimination against particular group of people")
    lst1 = []
    lst2= []
    lst3= []
    lst4= []
    dict1={}
    dict2={}
    for column in columns:
        priv_df = processed_data[processed_data[column]=='1.0']
        num_of_previleged = priv_df.shape[0]
        unpriv_df = processed_data[processed_data[column]=='0.0']
        num_of_unprevileged = unpriv_df.shape[0]
        unprivileged_outcomes = unpriv_df[unpriv_df['Labels']==1.0].shape[0]
        unprivileged_ratio = (unprivileged_outcomes/num_of_unprevileged)*100
        unprivileged_ratio = round(unprivileged_ratio,2)
    
        privileged_outcomes = priv_df[priv_df['Labels']==1.0].shape[0]
        privileged_ratio = (privileged_outcomes/num_of_previleged)*100
        privileged_ratio = round(privileged_ratio,2)
    
        label0 = processed_data[column].value_counts(normalize=True)['0.0']
        label0 = round(label0,2)*100
        label1 = processed_data[column].value_counts(normalize=True)['1.0']
        label1 = round(label1,2)*100
        
        lst1.append(unprivileged_ratio)
        lst2.append(privileged_ratio)
        lst3.append(label0)
        lst4.append(label1)
    dict1 = {'Unprevileged':lst1,'Privileged':lst2}
    dict2 = {'Label0':lst3,'Label1':lst4}
    
    #col1, col2 = st.beta_columns(2)
    col3, col4 = st.beta_columns(2)
    sex_plot = plot_bias(dict1,"Female","Male",'Unprevileged','Privileged',0,0)
    race_plot = plot_bias(dict1,"Black","White","Unprevileged","Privileged",1,1)
    orig_sex = plot_bias(dict2,"Female","Male","Label0",'Label1',0,0)
    orig_race = plot_bias(dict2,"White","Black","Label0",'Label1',1,1)
    #col1.header("Overall Percentage of Male/Female")
    #col1.plotly_chart(orig_sex,use_container_width=False, config={'displayModeBar': False})
    #col2.header("Overall Percentage of Black/White")
    #col2.plotly_chart(orig_race,use_container_width=False, config={'displayModeBar': False})
    col3.header("Percentage of Bias in Sex Column")
    col3.plotly_chart(sex_plot,use_container_width=False, config={'displayModeBar': False})
    col4.header("Percentage of Bias in Race Column")
    col4.plotly_chart(race_plot,use_container_width=False, config={'displayModeBar': False})
    st.markdown("Analysis Output")
    st.markdown("**Sex** Column shows higher percentage for male which indicates inclination of data towards Male compared to female.")
    st.markdown("**Race** Column shows higher percentage for whites which indicates inclination of data towards whites compared to blacks.")
    
    

def get_opt_pre_proc_dataset(dataset_name):
    dataset_info = {
        "Adult income dataset" : {
            "original_data_path" : r"C:\\Users\\Mahesh\\Desktop\\Hackathon\\CBA\\AIF_Adult_data_original.csv",
            "processed_data_path" : r"C:\\Users\\Joshika\\Desktop\\Navya\\CBA\\Bias Mitigation\\demo9\\AIF_final_adult_data_processed.csv",
            "description" : "The Adult income dataset classifies wheather an america adult will be earning more than $50K based on certain attributes\
                i.e age, education-num, sex, capital-gain, capital-loss, hours-per-week. Here sex is the protected attribute",
            "seed" : 119
        },
         "TW Credit Risk Dataset" : {
            "original_data_path" : r"D:\CBA_wok_docs\Work\Ethical_AI\Datasets\credit_card_default_taiwan\Streamlit_data\Taiwan_credit_default_data_original.csv",
            "processed_data_path" : r"D:\CBA_wok_docs\Work\Ethical_AI\Datasets\credit_card_default_taiwan\Streamlit_data\Taiwan_credit_default_data_processed.csv",
            "description" : "The TW Credit Risk Dataset classifies wheather an america adult will be earning more than $50K based on certain attributes\
                i.e age, education-num, sex, capital-gain, capital-loss, hours-per-week",
            "seed" : 11
        }
    }
    

    processed_data = pd.read_csv(dataset_info[dataset_name]["processed_data_path"])

    return processed_data   

def get_reweight_dataset(dataset_dict):
    if(dataset_dict['dataset_name'] == 'Adult income dataset'):
        if dataset_dict['protected_attribute_name'] == 'sex':
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
            dataset_orig = load_preproc_data_adult(['sex'])
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
            dataset_orig = load_preproc_data_adult(['race'])
    else:
        pass
    return dataset_orig

def get_dataset(dataset_dict):
    dataset_dict = {'dataset_name':'Adult income dataset',
                    'algo_name' : 'Reweighing',
                    'protected_attribute_name' : 'sex'}
    if(dataset_dict['algo_name'] == 'Optimized Pre-processing'):
        dataset_orig = get_opt_pre_proc_dataset(dataset_dict['dataset_name'])
    else:
        dataset_orig = get_reweight_dataset(dataset_dict)

    return dataset_orig

def load_dataset_UI():
    button_status = False
    dataset_dict = {}
    emp_st= st.empty()
    dataset_name = st.radio("Datasets on which analysis can be done",('Compas(ProPublica recidivism)', 'German credit scoring', 'Adult census income'))
    if(dataset_name == "Adult census income"):
        st.write("Classifies wheather an american adult will be earning more than $50K/year")
        data = pd.read_csv("C:\\Users\\Joshika\\Desktop\\Navya\\CBA\\Bias Mitigation\\demo9\\AIF_final_adult_data_processed.csv")
        st.write(data.head())
        
        
    elif(dataset_name == "Compas(ProPublica recidivism)"):
        st.write("Predict a criminal defendant’s likelihood of reoffending")


    elif(dataset_name == "German credit scoring"):
        st.write("Predict an individual's credit risk.")

    else:
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            #read csv
            dataset_dict=pd.read_csv(uploaded_file)

    return dataset_dict


def compare_results(original_value, repaired_value):
    result = ""
    if(original_value > repaired_value):
        result = "decreased"
    elif(original_value < repaired_value):
        result = "increased"
    else:
        result = "equal"
    return result


def key_points(acc_dict, sp_dict, di_dict):
    acc_result = compare_results(acc_dict['Original'], acc_dict['Repaired'])
    sp_result = compare_results(sp_dict['Original'], sp_dict['Repaired'])
    di_result = compare_results(di_dict['Original'], di_dict['Repaired'])

    if(acc_result == 'equal'):
        acc_statement = "Accuracy remains same for original and repaired data at "+str(acc_dict['Original'])
    else:
        acc_statement = "Accuracy "+acc_result+" from "+str(acc_dict['Original'])+" to "+str(acc_dict['Repaired'])

    if(sp_result == 'equal'):
        sp_statement = "Statistical parity remains same for original and repaired data at "+str(sp_dict['Original'])
    else:
        sp_statement = "Statistical parity "+sp_result+" from "+str(sp_dict['Original'])+" to "+str(sp_dict['Repaired'])

    if(di_result == 'equal'):
        di_statement = "Disparate impact remains same for original and repaired data at "+str(di_dict['Original'])
    else:
        di_statement = "Disparate impact "+di_result+" from "+str(di_dict['Original'])+" to "+str(di_dict['Repaired'])
    return acc_statement, sp_statement, di_statement




def plot_h_bar(data_dict, thrashold = 0, title = "Plot title"):
    bars = []
    bars.append(go.Bar(x=[data_dict["Original"]],
                    y=["Original"],
                    marker={'color': 'rgb(250,145,90)'},
                    name = "Original",
                    orientation = 'h',
                    width=[0.4,0.4,0.4],
                    ))

    bars.append(go.Bar(x=[data_dict["Repaired"]],
                    y=["Repaired"],
                    marker={'color': 'rgb(84, 194, 232)'},
                    name = "Repaired",
                    orientation = 'h',
                    width=[0.4,0.4,0.4]))

    
    fig = go.FigureWidget(data=bars)
    fig.update_layout(shapes=[
        dict(
        type= 'line',
        yref= 'paper', y0= 0, y1= 1,
        xref= 'x', x0= thrashold, x1= thrashold
        )
    ],
    title={
        'text': title,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
    )
    fig.update(layout_showlegend=False)
    fig.layout.height = 350
    fig.layout.width = 350
    return fig

def compare_page(dataset_name,protected_attribute,repair_value):
    processed_data = get_opt_pre_proc_dataset(dataset_name)
    seed = 119
    Y = processed_data["Labels"]
    X = processed_data.drop(["Labels"], axis = 1)
    column_names = list(processed_data.columns)
    X_train, y_train, X_test, y_test = train_test_split_reindexed(X, Y, seed)
    original_accuracy, original_pr_unpriv, original_pr_priv, \
    original_disparate_impact = logistic_regression_results(X_train, y_train, X_test, y_test,protected_column = protected_attribute)
    
    X_train_updated, y_train_updated, X_test_updated, y_test_updated = remove_disparate_impact_from_data(X_train, y_train,X_test, y_test,column_names,repair_level_per = repair_value,protected_attribute_name = protected_attribute)

    repaired_accuracy, repaired_pr_unpriv, repaired_pr_priv, \
    repaired_disparate_impact = logistic_regression_results(X_train_updated, \
                                                            y_train_updated, \
                                                            X_test_updated, y_test_updated,
                                                            protected_column = protected_attribute)

    original_statistical_parity = original_pr_unpriv - original_pr_priv
    repaired_statistical_parity = repaired_pr_unpriv - repaired_pr_priv
    acc_dict = {'Original' : round(original_accuracy,3)*100, 'Repaired' : round(repaired_accuracy,3)*100}
    di_dict = {'Original' : round(original_disparate_impact,3), 'Repaired' : round(repaired_disparate_impact,3)}
    sp_dict = {'Original' : round(original_statistical_parity,3), 'Repaired' : round(repaired_statistical_parity,3)}

    st.markdown("**Visualisation of Bias/Debias**")
    col1, col2 , col3= st.beta_columns(3)
    
    di_fig = plot_h_bar(di_dict, thrashold = 1,title = "Disparate Impact")
    sp_fig = plot_h_bar(sp_dict, thrashold = -0.001,title = "Statistical parity difference")
    acc_fig = plot_h_bar(acc_dict, thrashold = 75.2,title = "Accuracy")
    
    
    col1.plotly_chart(di_fig,use_container_width=False, config={'displayModeBar': False})
    col2.plotly_chart(sp_fig,use_container_width=False, config={'displayModeBar': False})
    col3.plotly_chart(acc_fig,use_container_width=False, config={'displayModeBar': False})
    

    acc_statement, sp_statement, di_statement = key_points(acc_dict, sp_dict, di_dict)
    
    st.markdown("** Results of the Analysis**")
    st.markdown(acc_statement)
    st.markdown(sp_statement)
    st.markdown(di_statement)

def load_data_pipeline():
    st.title("Data Bias Mitigation Demo")
    a = st.sidebar.empty()
    options =  ['Datasets for Analysis', 'Visualise Data Bias','Debias Algorithm', 'Debias Results']
    My_Page = a.radio('De-biasing work Flow', options)
    global Dataset_Dict1
    Dataset_Dict1 = {}
    if My_Page == 'Datasets for Analysis':
        Dataset_Dict1 = load_dataset_UI()
    elif My_Page == 'Debias Algorithm':
        Dataset_Dict1 = select_mitigation_algorithm(Dataset_Dict1)
    elif My_Page == 'Visualise Data Bias':
        Dataset_Dict1 = bias_check()
    else:
        dataset_name = "Adult income dataset"
        col1, col2 = st.beta_columns(2)
        protected_attribute = col1.selectbox("Protected Attributes",('sex','race'))
        repair_value = col2.slider("Bias Repair Level",0.0,1.0)
        compare_page(dataset_name,protected_attribute,repair_value)
    
if __name__ == "__main__":

    load_data_pipeline()