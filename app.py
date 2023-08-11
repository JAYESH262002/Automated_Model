from operator import index
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)


with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Automated Model")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Predict","Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    types_model = st.selectbox('Is It Regression or Classification Problem ?', ['Regression', 'Classification'])
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
        
        
    df2 = pd.get_dummies(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if 'chosen_target' not in st.session_state:
    st.session_state.chosen_target = None

if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df2.columns)
    if st.button('Run Modelling'): 
        setup(df2, target=chosen_target, verbose=False)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')
        
        # Store chosen_target in st.session_state
        st.session_state.chosen_target = chosen_target


if choice == "Predict":
    if st.session_state.chosen_target is None:
        st.warning("Please run the Modelling section first and choose the target column.")
    else:
        st.title("Predict New Data")
        model = load_model('best_model')
        chosen_target = st.session_state.chosen_target
        x = df.drop(chosen_target, axis=1)
        y = df[chosen_target]
        input_data = {}
        for col in df.columns:
            if col != chosen_target:
                input_data[col] = st.number_input(f"Enter value for {col}", value=0.0)

        types_model = st.selectbox('Model Is Of', ['Regression', 'Classification'])

        if st.button("Predict"):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
            
            if types_model == 'Regression':
                model = RandomForestRegressor()
            else:
                model = RandomForestClassifier()

            model.fit(x_train, y_train)
            
            # Convert input_data dictionary to a DataFrame
            input_df = pd.DataFrame([input_data])
            input_df = pd.get_dummies(input_df)
            
            # Make predictions on the input DataFrame
            y_preds = model.predict(input_df)
            
            st.write("Predicted Result:")
            st.write(y_preds)



if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")