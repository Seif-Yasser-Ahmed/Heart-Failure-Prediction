import streamlit as st
import mlflow
import pandas as pd
mlflow.set_tracking_uri("http://localhost:5000")

models = ['KNN', 'SVM', 'Logistic', 'DT', 'GaussBayes']
status = ['_No_Preprocessing', '_Preprocessing']
st.title(f"Mlflow Results")

for model in models:
    for state in status:
        experiment = model+state
        print(experiment)
        experiment = mlflow.get_experiment_by_name(experiment)
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        runs_df = pd.DataFrame(runs)
        runs_df = runs_df.drop(['run_id', 'experiment_id', 'status',
                               'artifact_uri', 'start_time', 'end_time', 'tags.mlflow.source.name', 'tags.mlflow.user', 'tags.mlflow.source.git.commit', 'tags.mlflow.source.type', 'tags.mlflow.log-model.history'], axis=1)
        cols = ['params.model', 'metrics.accuracy'] + \
            [col for col in runs_df.columns if col not in [
                'params.model', 'metrics.accuracy']]
        runs_df = runs_df[cols]
        st.subheader(f"Results for {model} with {state}")
        st.dataframe(runs_df)
