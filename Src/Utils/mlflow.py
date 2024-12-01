import mlflow


def log_models_to_mlflow(models, reports, exp_name):
    mlflow.set_experiment(exp_name)
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    for i, element in enumerate(models):
        model_name = element[0]
        model = element[1]
        report = reports[i]
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model", model_name)

            mlflow.log_metric('accuracy', report['accuracy'])

            mlflow.log_metric('recall_class_1', report['1']['recall'])
            mlflow.log_metric('recall_class_0', report['0']['recall'])
            mlflow.log_metric('recall_macro_avg',
                              report['macro avg']['recall'])

            mlflow.log_metric('precision_class_1', report['1']['precision'])
            mlflow.log_metric('precision_class_0', report['0']['precision'])
            mlflow.log_metric('precision_macro_avg',
                              report['macro avg']['precision'])

            mlflow.log_metric('f1_score_class_1', report['1']['f1-score'])
            mlflow.log_metric('f1_score_class_0', report['0']['f1-score'])
            mlflow.log_metric('f1_score_macro',
                              report['macro avg']['f1-score'])

            if "XGB" in model_name:
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
