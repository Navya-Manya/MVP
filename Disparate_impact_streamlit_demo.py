import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import random
import numpy as np
import plotly.graph_objects as go

from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

seed = random.randint(1, 10000)
np.random.seed(seed)


@st.cache(suppress_st_warning=True)
def load_data():
    dataset_dict = {
        "Adult Income Dataset" : {
            "original_data_path" : r"C:\Users\Joshika\Desktop\Navya\CBA\Bias Mitigation\demo9\AIF_final_adult_data_processed.csv",
            "processed_data_path" : r"C:\Users\Joshika\Desktop\Navya\CBA\Bias Mitigation\demo9\AIF_final_adult_data_processed.csv",
            #"description" : "The Adult income dataset classifies wheather an america adult will be earning more than $50K based on certain attributes\
               # i.e age, education-num, sex, capital-gain, capital-loss, hours-per-week. Here sex is the protected attribute",
           # "seed" : 119
        },
         "TW Credit Risk Dataset" : {
            "original_data_path" : r"D:\CBA_wok_docs\Work\Ethical_AI\Datasets\credit_card_default_taiwan\Streamlit_data\Taiwan_credit_default_data_original.csv",
            "processed_data_path" : r"D:\CBA_wok_docs\Work\Ethical_AI\Datasets\credit_card_default_taiwan\Streamlit_data\Taiwan_credit_default_data_processed.csv",
            #"description" : "The TW Credit Risk Dataset classifies wheather an america adult will be earning more than $50K based on certain attributes\
                #i.e age, education-num, sex, capital-gain, capital-loss, hours-per-week",
           # "seed" : 11
        }
    }
    
    #original_data = pd.read_csv(dataset_dict[dataset_name]["original_data_path"])
    #original_data = original_data.loc[:, ~original_data.columns.str.contains('^Unnamed')]
    #processed_data = pd.read_csv(dataset_dict[dataset_name]["processed_data_path"])
    #processed_data = processed_data.loc[:, ~processed_data.columns.str.contains('^Unnamed')]
    
    # file upload
    uploaded_file = st.file_uploader("File Upload")
    dataframe = pd.read_csv(uploaded_file)
    original_data = dataframe.loc[:, ~dataframe.columns.str.contains('^Unnamed')]
    processed_data = original_data.interpolate(limit_direction='both')
    return original_data, processed_data

def train_test_split_reindexed(X,y, seed = 119):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.34,random_state = seed)

    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, y_train, X_test, y_test

def calc_prop(data, group_col, group, output_col, output_val):
    new = data[data[group_col] == group]
    return len(new[new[output_col] == output_val])/len(new)


def logistic_regression_model(X_train, y_train):
    
    lg_model = LogisticRegression(class_weight='balanced', solver='liblinear')
    lg_model.fit(X_train, y_train)
    return lg_model

def logistic_regression_results(X_train, y_train,X_test, y_test,protected_column = "sex"):
    X_train = X_train.drop([protected_column], axis = 1)
    protected_data_x_test = list(X_test[protected_column])
    X_test = X_test.drop([protected_column], axis = 1)
    lg_model = logistic_regression_model(X_train, y_train)
    y_pred = lg_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    predicted_df = pd.DataFrame({protected_column: protected_data_x_test, "Labels": y_pred})
    pr_unpriv = calc_prop(predicted_df, protected_column, 0, "Labels", 1)
    pr_priv = calc_prop(predicted_df, protected_column, 1, "Labels", 1)
    disparate_impact = pr_unpriv / pr_priv

    return accuracy, pr_unpriv, pr_priv, disparate_impact

def gradient_boosting_model(X_train, y_train):
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=5, random_state=seed).fit(X_train, y_train)
    return gb_model

def gradient_boosting_results(X_train, y_train,X_test, y_test,protected_column = "sex"):
    X_train = X_train.drop([protected_column], axis = 1)
    protected_data_x_test = list(X_test[protected_column])
    X_test = X_test.drop([protected_column], axis = 1)
    gb_model = gradient_boosting_model(X_train, y_train)
    y_pred = gb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    predicted_df = pd.DataFrame({protected_column: protected_data_x_test, "Labels": y_pred})
    pr_unpriv = calc_prop(predicted_df, protected_column, 0, "Labels", 1)
    pr_priv = calc_prop(predicted_df, protected_column, 1, "Labels", 1)
    disparate_impact = pr_unpriv / pr_priv

    return accuracy, pr_unpriv, pr_priv, disparate_impact

def remove_disparate_impact_from_data(X_train, y_train,X_test, y_test,column_names, repair_level_per = 1.0, protected_attribute_name = "sex"):
    training_data = X_train.copy()
    training_data['Labels'] = list(y_train)

    testing_data = X_test.copy()
    testing_data['Labels'] = list(y_test)

    train_BLD = BinaryLabelDataset(favorable_label='1.0',
                                unfavorable_label='0.0',
                                df=training_data,
                                label_names=['Labels'],
                                protected_attribute_names=[protected_attribute_name],
                                unprivileged_protected_attributes=['0.0'])
    test_BLD = BinaryLabelDataset(favorable_label='1.0',
                                    unfavorable_label='0.0',
                                    df=testing_data,
                                    label_names=['Labels'],
                                    protected_attribute_names=[protected_attribute_name],
                                    unprivileged_protected_attributes=['0.0'])

    di = DisparateImpactRemover(repair_level=repair_level_per)
    rp_train = di.fit_transform(train_BLD)
    rp_test = di.fit_transform(test_BLD)

    rp_train_pd = pd.DataFrame(np.hstack([rp_train.features,rp_train.labels]),columns=column_names)
    rp_test_pd = pd.DataFrame(np.hstack([rp_test.features,rp_test.labels]),columns=column_names)

    y_train_updated = rp_train_pd['Labels']
    X_train_updated = rp_train_pd.drop(['Labels'],axis = 1)

    y_test_updated = rp_test_pd['Labels']
    X_test_updated = rp_test_pd.drop(['Labels'],axis = 1)

    return X_train_updated, y_train_updated, X_test_updated, y_test_updated


def reweighting(X_train, y_train,X_test, y_test,column_names, repair_level_per = 1.0, protected_attribute_name = "sex"):
    training_data = X_train.copy()
    training_data['Labels'] = list(y_train)

    testing_data = X_test.copy()
    testing_data['Labels'] = list(y_test)

    train_BLD = BinaryLabelDataset(favorable_label='1.0',
                                unfavorable_label='0.0',
                                df=training_data,
                                label_names=['Labels'],
                                protected_attribute_names=[protected_attribute_name],
                                unprivileged_protected_attributes=['0.0'])
    test_BLD = BinaryLabelDataset(favorable_label='1.0',
                                    unfavorable_label='0.0',
                                    df=testing_data,
                                    label_names=['Labels'],
                                    protected_attribute_names=[protected_attribute_name],
                                    unprivileged_protected_attributes=['0.0'])

    RW = Reweighing(unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)
    rp_train = RW.fit_transform(train_BLD)
    rp_test = RW.fit_transform(test_BLD)

    rp_train_pd = pd.DataFrame(np.hstack([rp_train.features,rp_train.labels]),columns=column_names)
    rp_test_pd = pd.DataFrame(np.hstack([rp_test.features,rp_test.labels]),columns=column_names)

    y_train_updated = rp_train_pd['Labels']
    X_train_updated = rp_train_pd.drop(['Labels'],axis = 1)

    y_test_updated = rp_test_pd['Labels']
    X_test_updated = rp_test_pd.drop(['Labels'],axis = 1)

    return X_train_updated, y_train_updated, X_test_updated, y_test_updated

def adversarial_debiasing(X_train, y_train,X_test, y_test,column_names, repair_level_per = 1.0, protected_attribute_name = "sex"):
    training_data = X_train.copy()
    training_data['Labels'] = list(y_train)

    testing_data = X_test.copy()
    testing_data['Labels'] = list(y_test)

    train_BLD = BinaryLabelDataset(favorable_label='1.0',
                                unfavorable_label='0.0',
                                df=training_data,
                                label_names=['Labels'],
                                protected_attribute_names=[protected_attribute_name],
                                unprivileged_protected_attributes=['0.0'])
    test_BLD = BinaryLabelDataset(favorable_label='1.0',
                                    unfavorable_label='0.0',
                                    df=testing_data,
                                    label_names=['Labels'],
                                    protected_attribute_names=[protected_attribute_name],
                                    unprivileged_protected_attributes=['0.0'])

    AD = AdversarialDebiasing(privileged_groups = privileged_groups,
                          unprivileged_groups = unprivileged_groups,
                          scope_name='plain_classifier',
                          debias=False,
                          sess=sess)
    rp_train = AD.fit_transform(train_BLD)
    rp_test = AD.fit_transform(test_BLD)

    rp_train_pd = pd.DataFrame(np.hstack([rp_train.features,rp_train.labels]),columns=column_names)
    rp_test_pd = pd.DataFrame(np.hstack([rp_test.features,rp_test.labels]),columns=column_names)

    y_train_updated = rp_train_pd['Labels']
    X_train_updated = rp_train_pd.drop(['Labels'],axis = 1)

    y_test_updated = rp_test_pd['Labels']
    X_test_updated = rp_test_pd.drop(['Labels'],axis = 1)

    return X_train_updated, y_train_updated, X_test_updated, y_test_updated



def run():

    


    def data_page():
        st.title("Data Bias Mitigation Demo")
        st.markdown("This data bias mitigation demo will help to discover and remove bias from data based on a protected attribute")
        original_data, processed_data,  = load_data()
        st.dataframe(original_data.head(10))
        
    
    def bias():
        original_data, processed_data,  = load_data()
        Y = processed_data["Labels"]
        X = processed_data.drop(["Labels"], axis = 1)
        X_train, y_train, X_test, y_test = train_test_split_reindexed(X, Y, seed)
        accuracy, pr_unpriv, pr_priv, disparate_impact = logistic_regression_results(X_train, y_train, X_test, y_test)
        st.write("Bias Present in Data : ",round(disparate_impact,2)*100)



    def results():
        original_data, processed_data,  = load_data()
        column_names = list(processed_data.columns)
        Y = processed_data["Labels"]
        X = processed_data.drop(["Labels"], axis = 1)
        X_train, y_train, X_test, y_test = train_test_split_reindexed(X, Y, seed)
        accuracy, pr_unpriv, pr_priv, disparate_impact = logistic_regression_results(X_train, y_train, X_test, y_test)
        st.markdown("**Results before Bias removal : **")
        st.write("Model Accuracy : ",round(accuracy,2)*100)
        st.write("Unprivileged proportion : ",round(pr_unpriv,2)*100)
        st.write("Privileged proportion : ",round(pr_priv,2)*100)
        st.write("Bias Present in Data : ",round(disparate_impact,2)*100)
    

        ## Starting the disparate impact removal pipeline

        X_train_updated, y_train_updated, X_test_updated, y_test_updated = remove_disparate_impact_from_data(X_train, y_train,X_test, y_test,column_names)

        accuracy, pr_unpriv, pr_priv, disparate_impact = gradient_boosting_results(X_train_updated, y_train_updated, X_test_updated, y_test_updated)
        st.markdown("**Results After Bias removal : **")
        st.write("Model Accuracy : ",round(accuracy,2)*100)
        st.write("Unprivileged proportion : ",round(pr_unpriv,2)*100)
        st.write("Privileged proportion : ",round(pr_priv,2)*100)
        st.write("Bias Present in Data : ",round(disparate_impact,2))

    page_names_to_funcs = {
"Data": data_page,
"Bias": bias,
"Results": results,
    }

    selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()

    
if __name__ == "__main__":
    run()

