import mlflow
from tqdm import tqdm
import warnings
from mlflow.models.signature import infer_signature


def log_models_to_mlflow(model_names, reports, exp_name):
    warnings.filterwarnings("ignore")
    mlflow.set_experiment(exp_name)
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    for i in tqdm(range(len(reports)), desc="Logging models to MLflow"):
        model_name = model_names[i]
        report = reports[model_name]
        model = report['model']
        X_test = report.get('X_test')
        y_test = report.get('y_test')

        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model", model_name)
            for parameter in report['model'].get_params():
                mlflow.log_param(
                    parameter, report['model'].get_params()[parameter])

            mlflow.log_metric('accuracy', report['accuracy'])

            mlflow.log_metric('recall_class_1', report['recall_class1'])
            mlflow.log_metric('recall_class_0', report['recall_class0'])

            mlflow.log_metric('precision_class_1', report['precision_class1'])
            mlflow.log_metric('precision_class_0', report['precision_class0'])

            mlflow.log_metric('f1_score_class_1', report['f1_class1'])
            mlflow.log_metric('f1_score_class_0', report['f1_class0'])

            if X_test is not None and y_test is not None:
                input_example = X_test[:5]
                signature = infer_signature(X_test, y_test)

                if "XGB" in model_name:
                    mlflow.xgboost.log_model(
                        model,
                        "model",
                        input_example=input_example,
                        signature=signature
                    )
                else:
                    mlflow.sklearn.log_model(
                        model,
                        "model",
                        input_example=input_example,
                        signature=signature
                    )
            else:
                if "XGB" in model_name:
                    mlflow.xgboost.log_model(model, "model")
                else:
                    mlflow.sklearn.log_model(model, "model")
