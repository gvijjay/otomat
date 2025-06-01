from dotenv import load_dotenv
# from .database import PostgreSQLDB
import base64
import hashlib
import os
import ast
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
import requests
import calendar
import joblib
# Import necessary libraries
import numpy as np
import pmdarima as pm
import tensorflow as tf
from digiotai.digiotai_jazz import Agent, Task, InputType, OutputType
from django.views.decorators.csrf import csrf_exempt
from dotenv import load_dotenv
from keras.models import load_model
from openai import OpenAI
from plotly.graph_objs import Figure
from prophet import Prophet
from rest_framework.decorators import api_view
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .database import PostgresDatabase, HanaDBManager

global connection_obj


# Fetch API key from Node.js API
def get_api_key():
    url = "https://otamat.com/api/get-token"  # Replace with actual API URL
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get("key")  # Extract the API key
        else:
            print("Error: Failed to fetch API key")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching API key: {e}")
        return None


# Configure OpenAI
load_dotenv()
OPENAI_API_KEY = get_api_key()

client = OpenAI(api_key=OPENAI_API_KEY)
expertise = "Interior Designer"
task = Task("Image Generation")
input_type = InputType("Text")
output_type = OutputType("Image")
agent = Agent(expertise, task, input_type, output_type)
api_key = OPENAI_API_KEY
name = "file_name"
headers = {
    'Authorization':
        'FlespiToken flespi_token'
}

db = PostgresDatabase()

os.makedirs('uploads', exist_ok=True)


def updatedtypes(df):
    datatypes = df.dtypes
    for col in df.columns:
        if datatypes[col] == 'object':
            try:
                pd.to_datetime(df[col])
                df.drop(col, axis=1, inplace=True)
                print(df.columns)
            except Exception as e:
                pass
    return df


def get_importance(X_train, y_train, model_type):
    if model_type == 'regression':
        model_ = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model_ = RandomForestClassifier(n_estimators=100, random_state=42)
    model_.fit(X_train, y_train)

    # Get feature importances
    feature_importance = model_.feature_importances_

    # Normalize feature importance to percentages
    feature_importance_percent = (feature_importance / np.sum(feature_importance)) * 100

    # Print feature importance scores in percentage
    df = pd.DataFrame({"Features": X_train.columns, "Importances": feature_importance_percent})
    df.sort_values(by='Importances', inplace=True, ascending=False)
    df.reset_index(inplace=True, drop=True)
    return df


def iscatcol(col, t, threshold=10):
    unique_values = col.dropna().unique()
    if len(unique_values) <= threshold or t == 'object':
        if t == 'object':
            return True, True  # Categorical and needs encoding
        return True, False  # Categorical but doesn't require encoding
    return False, False


def getcatcols(df):
    catcols = []
    catcols_encode = []
    unique_cols = {}
    for col in df.columns:
        a, b = iscatcol(df[col], df.dtypes[col])
        if a:
            catcols.append(col)
        if b:
            catcols_encode.append(col)
            unique_cols[col] = list(df[col].unique())
    return catcols, catcols_encode, unique_cols


def get_csv_metadata(df):
    metadata = {
        "columns": df.columns.tolist(),
        "data_types": df.dtypes.to_dict(),
        "null_values": df.isnull().sum().to_dict(),
        "example_data": df.head().to_dict()
    }
    return metadata


def data_cleanup(df):
    if 'Date' in df.columns and 'Time' in df.columns:
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df.drop(['Date', 'Time'], axis=1, inplace=True)  # Drop original columns if needed

    for col in df.select_dtypes(include='object').columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception:
            pass
    # 1. Removing columns with unique value
    for col in df.columns:
        if df[col].nunique() <= 5:
            print('dropping columns with single value', col)
            df.drop(col, axis=1, inplace=True)
    df = df.select_dtypes(include=['number', 'datetime'])
    # Selecting numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    # 2. Dropping low variance columns
    variances = df[numeric_cols].var()
    dynamic_variance_threshold = variances.median() * 0.1  # 10% of median variance
    low_variance_numeric = variances[variances < dynamic_variance_threshold].index.tolist()

    # variance_std_dev = variances.std()
    # variance_mean = variances.mean()
    #
    # # Define threshold as mean - 1 standard deviation
    # threshold = variance_mean - variance_std_dev
    #
    # # Identify low variance columns
    # low_variance_numeric = variances[variances < threshold].index.tolist()

    print('dropping low variance columns', low_variance_numeric)
    df.drop(low_variance_numeric, axis=1, inplace=True)
    return df


@csrf_exempt
def train_data(request, train_type, file_name):
    try:
        df = pd.read_csv(os.path.join("uploads", file_name.lower() + '.csv'))
        df = df.iloc[:300, :]
        if os.path.exists(os.path.join("data", file_name.lower())):
            return HttpResponse("Success")
        if train_type.lower() == 'predict':
            df = updatedtypes(df)
            os.makedirs(os.path.join("data", file_name.lower()), exist_ok=True)
            df.to_csv(os.path.join("data", file_name.lower(), "processed_data.csv"), index=False)
            for i in df.columns:
                try:
                    col_predict = i
                    print(col_predict)
                    label_encoders = {}
                    cat_col = False
                    # Split the data into features (X) and target variable (y)
                    X = df.drop(columns=[col_predict])
                    y = df[col_predict]
                    catcols, cat_cols_to_encode, unique_cols = getcatcols(X)
                    print(catcols, cat_cols_to_encode)
                    for column in cat_cols_to_encode:
                        label_encoders[column] = LabelEncoder()
                        X[column] = label_encoders[column].fit_transform(X[column])
                    dense_c = 1
                    if iscatcol(y, y.dtype)[0]:
                        label_encoders[col_predict] = LabelEncoder()
                        y = label_encoders[col_predict].fit_transform(y)
                        dense_c = len(label_encoders[col_predict].classes_)
                        cat_col = True
                    print(dense_c, "h")
                    scaler = StandardScaler()
                    numerical_features = list(set(X.columns) - set(catcols))
                    X[numerical_features] = scaler.fit_transform(X[numerical_features])
                    print(numerical_features)
                    model_type = None
                    # Split the data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = tf.keras.models.Sequential([
                        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                        tf.keras.layers.Dense(64, activation='relu')
                    ])
                    if iscatcol(df[col_predict], df.dtypes[col_predict])[0]:
                        # Define the architecture of the ANN model
                        model.add(tf.keras.layers.Dense(dense_c,
                                                        activation='softmax'))  # Output layer for binary classification
                        loss_function = 'sparse_categorical_crossentropy'
                        metrics = ['accuracy']
                        model_type = 'classification'

                    else:
                        model.add(tf.keras.layers.Dense(1))  # Output layer for regression
                        loss_function = 'mean_squared_error'
                        metrics = ['mae']
                        model_type = 'regression'
                    print(model_type)
                    # Compile the model
                    model.compile(optimizer='adam', loss=loss_function, metrics=metrics)

                    # Train the model
                    model.fit(X_train, y_train, epochs=500, batch_size=64, validation_split=0.1)

                    # Evaluate the model on the testing data
                    loss, accuracy = model.evaluate(X_test, y_test)
                    print("Test Accuracy:", accuracy)
                    if model_type == 'classification':
                        predictions = np.argmax(model.predict(X_test), axis=-1)
                    else:
                        predictions = model.predict(X_test)

                    for column in cat_cols_to_encode:
                        X_test[column] = label_encoders[column].inverse_transform(X_test[column])

                    if cat_col:
                        y_test = label_encoders[col_predict].inverse_transform(y_test)
                        predictions = label_encoders[col_predict].inverse_transform(predictions)

                    predicted_data = X_test.copy()
                    predicted_data["Actual"] = y_test
                    predicted_data["Predicted"] = predictions
                    print(predicted_data.head(5))
                    # Sort the DataFrame by importance
                    importance_df = get_importance(X_train, y_train, model_type)

                    # Print or plot the top features
                    print(importance_df)
                    if not os.path.exists(os.path.join("data", file_name.lower(), col_predict.replace(" ", "_"))):
                        os.makedirs(os.path.join("data", file_name.lower(), col_predict.replace(" ", "_")))

                    predicted_data.to_csv(
                        os.path.join("data", file_name.lower(), col_predict.replace(" ", "_"), 'predictions.csv'),
                        index=False)
                    # Save the label encoders
                    for column, encoder in label_encoders.items():
                        joblib.dump(encoder,
                                    os.path.join("data", file_name.lower(), col_predict.replace(" ", "_"),
                                                 f'{column.replace(" ", "_")}_encoder.pkl'))

                    # Save the trained model
                    print("saved_path",
                          os.path.join("data", file_name.lower(), col_predict.replace(" ", "_"), "model.h5"))
                    model.save(os.path.join("data", file_name.lower(), col_predict.replace(" ", "_"), "model.h5"))
                    if model_type == 'classification':
                        accuracy = accuracy * 100
                        metrics = "Accuracy"
                    else:
                        metrics = "MAE"

                    results = {
                        "accuracy": accuracy,
                        "metrics": metrics,
                        "Top Fields": importance_df.to_json(),
                        "Plot": [int(value) for value in X_train[importance_df[importance_df.columns[0]][0]].values],
                        "sample_rows": predicted_data.to_json()
                    }
                    cols = {c: unique_cols[c] if c in unique_cols else None for c in X_train.columns}

                    with open(os.path.join("data", file_name.lower(), col_predict.replace(" ", "_"), "deployment.json"),
                              "w") as fp:
                        json.dump({"columns": cols, "model_type": model_type}, fp, indent=4)
                    with open(os.path.join("data", file_name.lower(), col_predict.replace(" ", "_"), "results.json"),
                              "w") as fp:
                        json.dump(results, fp, indent=4)
                except Exception as e:
                    print(e)
                    return HttpResponse("Error " + str(e))
            return HttpResponse('Success')
        elif train_type.lower() == 'forecast':
            df = data_cleanup(df)
            data = df
            try:
                numeric_cols = df.select_dtypes(include=['datetime']).columns.tolist()
                date_column = numeric_cols[0]
                if not date_column:
                    raise ValueError("No datetime column found in the dataset.")
                print(date_column)
                # Set the date column as index
                data[date_column] = pd.to_datetime(data[date_column])
                data.set_index(date_column, inplace=True)
                print(data.head(15))
                # Identify forecast columns (numeric columns)
                forecast_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                if not forecast_columns:
                    raise ValueError("No numeric columns found for forecasting in the dataset.")

                time_differences = data.index.to_series().diff().dropna()
                print(time_differences)
                inconsistent_intervals = time_differences[time_differences != time_differences.mode()[0]]
                print(inconsistent_intervals)

                # Infer frequency of datetime index
                freq = pd.infer_freq(data.index)
                print(date_column, freq)
                # Determine m based on inferred frequency
                if freq == '5T':  # Five-minute data
                    m = 288  # Daily seasonality (288 intervals in a day)
                elif freq == '15T':  # Quarter-hourly data (every 15 minutes)
                    m = 96  # Daily seasonality (96 intervals in a day)
                elif freq == '30T':  # Half-hourly data (every 30 minutes)
                    m = 48  # Daily seasonality (48 intervals in a day)
                elif freq == 'H':  # Hourly data
                    m = 24  # Daily seasonality (24 intervals in a day)
                elif freq == 'D':  # Daily data
                    m = 7  # Weekly seasonality (7 days in a week)
                elif freq == 'W':  # Weekly data
                    m = 52  # Yearly seasonality (52 weeks in a year)
                elif freq == 'M':  # Monthly data
                    m = 12  # Yearly seasonality (12 months in a year)
                elif freq == 'Q':  # Quarterly data
                    m = 4  # Yearly seasonality (4 quarters in a year)
                elif freq == 'A':  # Annual data
                    m = 1  # No further seasonality within a year
                else:
                    raise ValueError(
                        f"Unsupported frequency '{freq}'. Ensure data is in a common time interval.")

                results = {}
                for col in forecast_columns:
                    try:
                        data_actual = data[col].dropna()  # Remove NaNs if any

                        # Split data into train and test sets
                        train = data_actual.iloc[:-m]
                        test = data_actual.iloc[-m:]

                        # Auto ARIMA model selection
                        model = pm.auto_arima(train,
                                              m=m,  # frequency of seasonality
                                              seasonal=True,  # Enable seasonal ARIMA
                                              d=None,  # determine differencing
                                              test='adf',  # adf test for differencing
                                              start_p=0, start_q=0,
                                              max_p=12, max_q=12,
                                              D=None,  # let model determine seasonal differencing
                                              trace=True,
                                              error_action='ignore',
                                              suppress_warnings=True,
                                              stepwise=True)

                        # Forecast and calculate errors
                        fc, confint = model.predict(n_periods=m, return_conf_int=True)
                        # Save results to dictionary
                        results = {
                            "actual": {
                                "date": list(test.index.astype(str)),
                                "values": [float(val) if isinstance(val, np.float_) else int(val) for val in
                                           test.values]
                            },
                            "forecast": {
                                "date": list(test.index.astype(str)),
                                "values": [float(val) if isinstance(val, np.float_) else int(val) for val in fc]
                            }
                        }
                        if not os.path.exists(os.path.join("data", file_name.lower(), col.replace(" ", "_"))):
                            os.makedirs(os.path.join("data", file_name.lower(), col.replace(" ", "_")), exist_ok=True)
                        col = col.replace(" ", "_")
                        with open(os.path.join('data', file_name.lower(), col,
                                               col.lower() + '_results.json'), 'w') as fp:
                            json.dump(results, fp)
                        print(
                            f"Results saved to {os.path.join('data', file_name.lower(), col, col.lower() + '_results.json')}")
                    except Exception as e:
                        print(e)
                        return HttpResponse("Error " + str(e))
                return HttpResponse("Success")
            except Exception as e:
                print(e)
                return HttpResponse("Error " + str(e))
    except Exception as e:
        print(e)
        return HttpResponse("Error " + str(e))


# Database connection
@csrf_exempt
def connection(request):
    global connection_obj
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        database = request.POST['database']
        host = request.POST['host']
        port = request.POST['port']
        connection_obj = db.create_connection(username, password, database, host, port)
        print(connection_obj)
        return HttpResponse(json.dumps({"tables": connection_obj}), content_type="application/json")


# Upload functionality only
import os
import json
import io
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.exceptions import SuspiciousOperation


@csrf_exempt
def upload_and_store_data(request):
    try:
        print("[DEBUG] Received a request with method:", request.method)  # Debug statement

        if request.method == "POST":
            email = request.POST.get("mail")
            files = request.FILES.get("file")  # Retrieve the uploaded file

            if not files:
                return JsonResponse({"error": "No files uploaded"}, status=400)

            file_name = files.name
            file_extension = os.path.splitext(file_name)[1].lower()  # Extract file extension

            try:
                # Create a directory for storing uploaded files
                upload_dir = "uploads"
                os.makedirs(upload_dir, exist_ok=True)

                # Save the uploaded file locally
                local_file_path = os.path.join(upload_dir, file_name)
                with open(local_file_path, "wb") as f:
                    for chunk in files.chunks():
                        f.write(chunk)

                # Process the uploaded file based on its extension
                if file_extension == ".csv":
                    print("[DEBUG] Processing as CSV file...")  # Debug statement
                    files.seek(0)  # Reset file pointer
                    content = files.read().decode("utf-8")
                    csv_data = io.StringIO(content)
                    df = pd.read_csv(csv_data)
                    print("[DEBUG] CSV parsed successfully. DataFrame shape:", df.shape)  # Debug statement

                elif file_extension in [".xls", ".xlsx"]:
                    print("[DEBUG] Processing as Excel file...")  # Debug statement
                    df = pd.read_excel(local_file_path)  # Read from the saved local file
                    print("[DEBUG] Excel parsed successfully. DataFrame shape:", df.shape)  # Debug statement

                else:
                    files.seek(0)  # Reset file pointer
                    content = files.read().decode("utf-8")
                    csv_data = io.StringIO(content)
                    df = pd.read_csv(csv_data)

                # Validate the DataFrame
                if df.empty:
                    return JsonResponse({"error": "Uploaded file contains no data"}, status=400)

                csv_file_path = os.path.join(upload_dir, file_name.replace(file_extension, '.csv').lower())
                df.to_csv(csv_file_path, index=False)

                excel_file_path = os.path.join(upload_dir, file_name.replace(file_extension, '.xlsx').lower())
                df.to_excel(excel_file_path, index=False, engine='openpyxl')

                df.to_csv('data.csv', index=False)
                df.to_excel('data1.xlsx', index=False, engine='openpyxl')

                print("upload_and_store_data........", df.head(3))

                # Store data directly into the database
                print("[DEBUG] Storing data into the database...")  # Debug statement
                results = db.insert_or_update(email, df, file_name)  # Insert into MongoDB
                print("[DEBUG] Database operation results:", results)  # Debug statement

                # Prepare response
                response_data = {
                    "message": "File uploaded, stored locally, and data saved to the database successfully",
                    "upload_status": results,
                    "preview": df.head(10).to_dict(orient="records"),
                }

                return JsonResponse(response_data, safe=False)

            except Exception as e:
                print("[ERROR] Failed to process and store file:", str(e))  # Debug statement
                return JsonResponse({"error": f"Failed to process and store file: {str(e)}"}, status=500)

        elif request.method == "GET":
            print("[DEBUG] GET method is not supported for this endpoint")  # Debug statement
            return JsonResponse({"error": "GET method is not supported for this endpoint"}, status=405)

        print("[DEBUG] Invalid request method")  # Debug statement
        return JsonResponse({"error": "Invalid Request Method"}, status=405)

    except Exception as e:
        print("[ERROR] An error occurred:", str(e))  # Debug statement
        return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)


@csrf_exempt
def upload_and_analyze_data(request):
    try:
        if request.method == 'POST':
            # Load the previously stored email and data

            csv_file_path = 'data.csv'
            df = pd.read_csv(csv_file_path)
            print(df.head(5))

            kpi_file = request.FILES.get("kpi_file")
            file_name = os.path.basename(csv_file_path)
            print(file_name)

            # Analyze and process the data
            data_file_name, kpi_config_file_name = file_name, ''

            new_df, html_df = process_missing_data(df.copy())
            cache.set('dataframe', html_df)
            request.session['dataframe'] = html_df

            new_df.to_csv(os.path.join('uploads', 'processed_data.csv'), index=False)

            if os.path.exists('kpis.json'):
                os.remove('kpis.json')
            request.session['uploadedFileName'] = file_name

            if kpi_file:
                kpis_dict = xmltodict.parse(kpi_file.read())
                with open('uploads/kpi_config.json', 'w') as json_file:
                    json.dump(kpis_dict, json_file, indent=4)
                    kpi_config_file_name = kpi_file.name

            with open('uploads/configs.json', 'w') as json_file:
                json.dump({
                    "data_file_name": data_file_name,
                    "kpi_config_file_name": kpi_config_file_name
                }, json_file, indent=4)

            # Run data analysis
            response_data1 = analyze_data(df)
            if isinstance(response_data1, pd.DataFrame):
                response_data1 = response_data1.to_dict(orient='records')

            response_data1['preview'] = df.head(10).to_dict(orient='records')
            response_data1['upload_status'] = "Data reused successfully"

            return JsonResponse(response_data1, safe=False)

        elif request.method == 'GET':
            # Load and serve processed data
            upload_dir = "uploads"
            processed_file_path = os.path.join(upload_dir, "processed_data.csv")
            if os.path.exists(processed_file_path):
                processed_df = pd.read_csv(processed_file_path)
                return JsonResponse({"df_preview": processed_df.head(10).to_dict(orient='records')})

            return JsonResponse({"error": "No processed data found"}, status=400)

        return JsonResponse({"error": "Invalid Request Method"}, status=405)

    except Exception as e:
        return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)


# Upload data to the database (CSV and Excel)
import os
import io
import shutil
import json
import pandas as pd
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import xmltodict
from django.core.cache import cache
from django.core.exceptions import SuspiciousOperation


# Original
# @csrf_exempt
# def upload_and_analyze_data(request):
#     try:
#         if request.method == 'POST':
#             email = request.POST.get('mail')
#             files = request.FILES.get('file')  # Retrieve the uploaded file
#             kpi_file = request.FILES.get("kpi_file")
#
#             if not files:
#                 return JsonResponse({"error": "No files uploaded"}, status=400)
#
#             file_name = files.name
#             file_extension = os.path.splitext(file_name)[1].lower()  # Extract file extension
#
#             try:
#                 # Process the uploaded file based on its extension
#                 if file_extension == '.csv':
#                     content = files.read().decode('utf-8')
#                     csv_data = io.StringIO(content)
#                     df = pd.read_csv(csv_data)
#                 elif file_extension in ['.xls', '.xlsx']:
#                     df = pd.read_excel(files)
#                 else:
#                     raise SuspiciousOperation("Unsupported file format")
#
#                 # Save the uploaded file locally for backup/logging purposes
#                 upload_dir = "uploads"
#                 os.makedirs(upload_dir, exist_ok=True)
#
#                 csv_file_path = os.path.join(upload_dir, file_name.replace(file_extension, '.csv').lower())
#                 df.to_csv(csv_file_path, index=False)
#
#                 excel_file_path = os.path.join(upload_dir, file_name.replace(file_extension, '.xlsx').lower())
#                 df.to_excel(excel_file_path, index=False, engine='openpyxl')
#
#                 df.to_csv('data.csv', index=False)
#                 df.to_excel('data1.xlsx', index=False, engine='openpyxl')
#
#                 results = db.insert_or_update(email, df, file_name)  # Insert into MongoDB
#
#                 data_file_name, kpi_config_file_name = file_name, ''
#
#                 new_df, html_df = process_missing_data(df.copy())
#                 cache.set('dataframe', html_df)
#                 request.session['dataframe'] = html_df
#
#                 new_df.to_csv(os.path.join('uploads', 'processed_data.csv'), index=False)
#
#                 if os.path.exists('kpis.json'):
#                     os.remove('kpis.json')
#                 request.session['uploadedFileName'] = files.name
#
#                 if kpi_file:
#                     kpis_dict = xmltodict.parse(kpi_file.read())
#                     with open('uploads/kpi_config.json', 'w') as json_file:
#                         json.dump(kpis_dict, json_file, indent=4)
#                         kpi_config_file_name = kpi_file.name
#
#                 with open('uploads/configs.json', 'w') as json_file:
#                     json.dump({
#                         "data_file_name": data_file_name,
#                         "kpi_config_file_name": kpi_config_file_name
#                     }, json_file, indent=4)
#
#                 response_data1 = analyze_data(df)  # Assuming analyze_data is a function that analyzes data
#                 response_data1['preview'] = df.head(10).to_dict(orient='records')
#                 response_data1['upload_status'] = results
#
#                 return JsonResponse(response_data1, safe=False)
#
#             except Exception as e:
#                 return JsonResponse({"error": f"Failed to upload and analyze file: {str(e)}"}, status=500)
#
#         elif request.method == 'GET':
#             if os.path.exists('uploads') and os.path.exists(os.path.join('uploads', 'data.csv')):
#                 data_frame = pd.read_csv(os.path.join('uploads', 'data.csv'))
#                 data_frame = updatedtypes(data_frame)
#
#                 with open('uploads/configs.json', 'r') as json_file:
#                     data = json.load(json_file)
#                     return JsonResponse({
#                         "uploadedInfo": True,
#                         "df_preview": data_frame.head().to_dict(orient='records'),
#                         "data_file_name": data['data_file_name'],
#                         "kpi_config_file_name": data["kpi_config_file_name"]
#                     })
#
#             return JsonResponse({"uploadedInfo": False})
#
#         return JsonResponse({"error": "Invalid Request Method"}, status=405)
#
#     except Exception as e:
#         return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)

def analyze_data(df):
    # Extract the first 10 rows of the data
    first_10_rows = df.head(10).to_dict(orient='records')
    # Generate descriptions and questions for the data based on table name and columns
    columns_list = ", ".join(df.columns)
    text_questions = {}
    plotting_questions = {}
    forecasting_questions = {}
    prompt_eng = (
        f"You are analytics_bot. Analyse the data: {df.head()} and give description of the columns"
    )
    column_description = generate_code(prompt_eng)
    trials = 3
    while trials > 0:
        try:
            prompt_eng1 = (
                    f"Based on the data with sample records as {df.head()}, generate 5 questions based on data." + "output should be in the format like  {'question1':...., 'question2':...., so on..} ")
            text_questions = generate_code(prompt_eng1)
            text_questions = ast.literal_eval(text_questions)
            break
        except Exception as e:
            print(e)
        trials -= 1
    trials = 3
    while trials > 0:
        try:
            prompt_eng_2 = f"Based on the data with sample records as {df.head()}, " + "Generate 5 plotting questions based on data, Each question should start with plot keyword. output should be in the format like  {'question1':...., 'question2':...., so on..}"
            plotting_questions = generate_code(prompt_eng_2)
            plotting_questions = ast.literal_eval(plotting_questions)
            break
        except Exception as e:
            print(e)
        trials -= 1
    trials = 3
    while trials > 0:
        try:
            # Creating the forecasting questions
            prompt_eng_3 = (
                # f"Generate 5 forecasting questions for the data: {df}"

                f"Using the dataset {df}, generate 5 forecasting-related questions based on the dataset. "
                f"The questions should: "
                f"1. Start with the word **'Forecast'**. "
                f"2. The questions should be very simple and straight forward."
                f"3. Design the questions to give visually interpretable outputs, such as charts or graphs, for forecasting analysis."
                f"4. Examples: 'Forecast the sales trend for the next 6 months' or 'Forecast the quarterly revenue growth for the next year.'"
                "5. output should be in the format like  {'question1':...., 'question2':...., so on..} "

                # f"Given the dataset {df}, generate 5 forecasting-related questions that meet the following criteria: "
                # f"1. The questions should be **specific**, **realistic**, and **focused on measurable metrics or trends**. "
                # f"2. Each question should **start with 'Forecast'** and address **clear forecasting goals**. "
                # f"3. Tailor the questions to the type of data in the dataset, ensuring they align with the trends, patterns, or key variables observed in the data."

                # f"Using the dataset {df}, generate 5  forecasting related questions. "
                # f"Each question must start with 'Forecast' and focus on predicting measurable trends within the dataset. "
                # f"Design the questions to give visually interpretable outputs, such as charts or graphs, for forecasting analysis. "
                # f"Examples: 'Forecast the sales trend for the next 6 months' or 'Forecast the quarterly revenue growth for the next year.' "
            )

            forecasting_questions = generate_code(prompt_eng_3)

            forecasting_questions = ast.literal_eval(forecasting_questions)
            break
        except Exception as e:
            print(e)
        trials -= 1
    # Create a JSON response with titles corresponding to each prompt
    response_data = {
        "all_records": df.to_dict(orient='records'),
        "first_10_rows": first_10_rows,  # Include first 10 rows
        "column_description": column_description,
        "text_questions": text_questions,
        "plotting_questions": plotting_questions,
        "forecasting_questions": forecasting_questions
    }
    return response_data


def process_missing_data(df):
    df = convert_to_datetime(df)
    df, html_df = handle_missing_data(df)
    return df, html_df


def convert_to_datetime(df):
    # Define the possible date formats to try
    date_formats = ['%m-%d-%Y', '%m/%d/%Y', '%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%Y/%m/%d']

    # Loop through each column
    for col in df.columns:
        # Only process object columns, assuming they may contain dates in string format
        if df[col].dtype == 'object':
            # Check if the column contains potential date strings
            if df[col].str.contains(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', na=False).any():
                # Try to parse automatically first
                try:
                    df[col] = pd.to_datetime(df[col], errors='raise')
                except (ValueError, TypeError):
                    # If automatic parsing fails, try each format individually
                    def parse_date(value):
                        for fmt in date_formats:
                            try:
                                return pd.to_datetime(value, format=fmt)
                            except (ValueError, TypeError):
                                continue
                        return pd.NaT  # Return NaT if none of the formats match

                    # Apply the custom parse function to handle multiple formats
                    df[col] = df[col].apply(parse_date)
    return df


from sklearn.impute import KNNImputer


def handle_missing_data(df):
    try:
        # Identify numeric and datetime columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        date_time_cols = df.select_dtypes(include=['datetime64']).columns

        # Impute numeric columns and track which cells were imputed
        imputer = KNNImputer(n_neighbors=5)
        imputed_numeric = imputer.fit_transform(df[numeric_cols])
        imputed_numeric_df = pd.DataFrame(imputed_numeric, columns=numeric_cols)

        # Mark imputed cells (True if the original cell was NaN)
        imputed_flags = df[numeric_cols].isnull()
        imputed_flags = imputed_flags.applymap(lambda x: x if x else False)

        # Update DataFrame with imputed values
        df[numeric_cols] = imputed_numeric_df

        # Handle datetime columns by forward filling missing values
        for col in date_time_cols:
            df[col] = pd.to_datetime(df[col])
            time_diffs = df[col].diff().dropna()
            avg_diff_sec = time_diffs.mean().total_seconds()
            minute_sec = 60
            hour_sec = 3600
            day_sec = 86400
            month_sec = day_sec * 30.44
            year_sec = day_sec * 365.25

            if avg_diff_sec < hour_sec:
                time_unit = "minutes"
                avg_diff = pd.Timedelta(minutes=avg_diff_sec / minute_sec)
            elif avg_diff_sec < day_sec:
                time_unit = "hours"
                avg_diff = pd.Timedelta(hours=avg_diff_sec / hour_sec)
            elif avg_diff_sec < month_sec:
                time_unit = "days"
                avg_diff = pd.Timedelta(days=avg_diff_sec / day_sec)
            elif avg_diff_sec < year_sec:
                time_unit = "months"
                avg_diff = pd.DateOffset(months=round(avg_diff_sec / month_sec))
            else:
                time_unit = "years"
                avg_diff = pd.DateOffset(years=round(avg_diff_sec / year_sec))

            for i in range(1, len(df)):
                if pd.isnull(df[col].iloc[i]):
                    df.loc[i, col] = df[col].iloc[i - 1] + avg_diff
                    imputed_flags.loc[i, col] = True

            imputed_flags.fillna(False, inplace=True)

        # Convert the DataFrame into a JSON-serializable format with flags
        data = []
        for _, row in df.iterrows():
            row_data = {}
            for col in df.columns:
                row_data[col] = {
                    "value": row[col].strftime('%Y-%m-%d %H:%M:%S') if isinstance(row[col], pd.Timestamp) else row[col],
                    "is_imputed": str(imputed_flags[col].get(_, False)) if col in imputed_flags else str(False)
                    # Check if cell was imputed
                }
            data.append(row_data)
        return df, data
    except Exception as e:
        print(e)


def serialize_datetime(obj):
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError("Type not serializable")


# Showing the number of tables in the database
@csrf_exempt
def get_tableinfo(request):
    if request.method == 'POST':
        table_info = db.get_tables_info()
        return HttpResponse(table_info, content_type="application/json")


@csrf_exempt
def get_user_data(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        table_info = db.get_user_tables(email)
        print(table_info)
        return HttpResponse(json.dumps({"result": table_info}), content_type="application/json")


# Deleting all the tables based on the user email
@csrf_exempt
def delete_all_user_tables(request):
    if request.method == 'POST':
        try:
            email = request.POST.get('email')
            print(f"Received email for deletion: {email}")  # Debug statement

            if not email:
                print("Email is missing in the request.")  # Debug statement
                return JsonResponse({"error": "Email is required"}, status=400)

            # Assuming `db` is an instance of your database class
            print(f"Calling delete_user_tables method with email: {email}")  # Debug statement
            deletion_status = db.delete_all_tables_data(email)

            if deletion_status:
                print(f"All tables associated with email '{email}' have been deleted.")  # Debug statement
                return JsonResponse({"message": f"All tables associated with email '{email}' have been deleted."},
                                    status=200)
            else:
                print(f"No tables found for email '{email}' or an error occurred.")  # Debug statement
                return JsonResponse({"error": f"No tables found for email '{email}' or an error occurred."}, status=404)
        except Exception as e:
            print(f"Exception occurred while deleting tables: {str(e)}")  # Debug statement
            return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)
    print("Invalid request method. Only POST is allowed.")  # Debug statement
    return JsonResponse({"error": "Invalid request method. Use POST."}, status=405)


# Deleting the user-specific list of tables
@csrf_exempt
def delete_selected_tables_by_name(request):
    if request.method == 'POST':
        try:
            # Extract user email and table names from the request
            print("Parsing request body for table deletion.")  # Debug statement

            email = request.POST.get('email')
            table_names = request.POST.getlist('table_names')  # List of table names to delete

            print(f"Received email: {email}, table names: {table_names}")  # Debug statement

            if not email or not table_names:
                print("Missing 'email' or 'table_names' in the request.")  # Debug statement
                return JsonResponse({"error": "Both 'email' and 'table_names' are required"}, status=400)

            if not isinstance(table_names, list):
                print(f"Invalid data type for table_names: {type(table_names)}")  # Debug statement
                return JsonResponse({"error": "'table_names' must be a list"}, status=400)

            # Assuming `db` is your database manager instance
            print(
                f"Calling delete_selected_user_tables_by_name with email: {email} and table_names: {table_names}")  # Debug statement
            deletion_status = db.delete_tables_data(email, table_names)

            if deletion_status:
                print(f"Deleted {len(table_names)} table(s) for email '{email}'.")  # Debug statement
                return JsonResponse({
                    "message": f"{len(table_names)} table(s) associated with email '{email}' have been deleted."
                }, status=200)
            else:
                print(
                    f"No matching tables found for email '{email}' or the provided table names: {table_names}")  # Debug statement
                return JsonResponse({
                    "error": f"No matching tables found for email '{email}' or the provided table names."
                }, status=404)
        except Exception as e:
            print(f"Exception occurred while deleting selected tables: {str(e)}")  # Debug statement
            return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)
    print("Invalid request method. Only POST is allowed.")  # Debug statement
    return JsonResponse({"error": "Invalid request method. Use POST."}, status=405)


# Showing the data to the user based on the table name
@csrf_exempt
def read_db_table_data(request):
    if request.method == 'POST':
        tablename = request.POST['tablename']
        df = db.get_table_data(tablename)
        df.to_csv('data.csv', index=False)
        df.to_csv(os.path.join("uploads", tablename.lower() + '.csv'), index=False)
        response_data = analyze_data(df)
        return JsonResponse(response_data, safe=False)
        # print(response_data)
        # return HttpResponse(json.dumps({"result": response_data}, default=serialize_datetime),
        #                     content_type="application/json")


@csrf_exempt
def read_data(request):
    if request.method == 'POST':
        tablename = request.POST['tablename']
        df = db.get_table_data(tablename)
        df.to_csv('data.csv', index=False)
        df.to_csv(os.path.join("uploads", tablename.lower() + '.csv'), index=False)
        return HttpResponse(df.to_json(), content_type="application/json")


# Function to generate code from OpenAI API
def generate_code(prompt_eng):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_eng}
        ]
    )
    all_text = ""
    for choice in response.choices:
        message = choice.message
        chunk_message = message.content if message else ''
        all_text += chunk_message
    print(all_text)
    if "```python" in all_text:
        code_start = all_text.find("```python") + 9
        code_end = all_text.find("```", code_start)
        code = all_text[code_start:code_end]
    else:
        code = all_text
    return code


@csrf_exempt
def regenerate_txt(request):
    if request.method == "POST":
        df = pd.read_csv('data.csv')
        prompt_eng = (
                f"Based on the data with sample records as {df.head()}, generate 5 questions based on data." + " output should be in the format like  {'question1':...., 'question2':...., so on..}"
        )
        text_questions = {}
        trials = 3
        while trials > 0:
            try:
                text_questions = generate_code(prompt_eng)
                text_questions = ast.literal_eval(text_questions)
                break
            except Exception as e:
                print(e)
            trials -= 1
        return HttpResponse(json.dumps({"questions": text_questions}),
                            content_type="application/json")


@csrf_exempt
def regenerate_chart(request):
    if request.method == "POST":
        df = pd.read_csv('data.csv')
        prompt_eng = (
                f"Based on the data with sample records as {df.head()}. Generate 5 plotting questions based on data. question shoud start with plot keyword" + " output should be in the format like  {'question1':...., 'question2':...., so on..}"
        )
        code = {}
        trials = 3
        while trials > 0:
            try:
                code = generate_code(prompt_eng)
                code = ast.literal_eval(code)
                break
            except Exception as e:
                print(e)
            trials -= 1
        return HttpResponse(json.dumps({"questions": code}),
                            content_type="application/json")


# For Forecast
@csrf_exempt
def regenerate_forecast(request):
    if request.method == "POST":
        df = pd.read_csv('data.csv')
        prompt_eng = (
                f"Using the dataset {df}, Regenerate 5 forecasting-related questions based on the dataset. "
                f"The questions should: "
                f"1. Start with the word **'Forecast'**. "
                f"2. The questions should be very simple and straight forward."
                f"3. Design the questions to give visually interpretable outputs, such as charts or graphs, for forecasting analysis."
                f"4. Examples: 'Forecast the sales trend for the next 6 months' or 'Forecast the quarterly revenue growth for the next year.'"
                + "5. output should be in the format like  {'question1':...., 'question2':...., so on..}"
        )
        trials = 3
        code = {}
        while trials > 0:
            try:
                code = generate_code(prompt_eng)
                code = ast.literal_eval(code)
                break
            except Exception as e:
                print(e)
            trials -= 1
        return HttpResponse(json.dumps({"questions": code}),
                            content_type="application/json")


@csrf_exempt
def gen_txt_response(request):
    if request.method == "POST":
        csv_file_path = 'data.csv'
        df = pd.read_csv(csv_file_path)
        # Generate CSV metadata
        csv_metadata = {"columns": df.columns.tolist()}
        metadata_str = ", ".join(csv_metadata["columns"])
        query = request.POST["query"]
        prompt_eng = (
            f"""
                You are a Python expert focused on answering user queries about data preprocessing. Always strictly adhere to the following rules:               
                1. Generic Queries:
                    If the user's query is generic and not related to data, respond with a concise and appropriate print statement. For example:

                    Query: "What is AI?"
                    Response: "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines."
                2. Data-Related Queries:
                    If the query is about data processing, assume the file data.csv is the data source and contains the following columns: {metadata_str}.

                    For these queries, respond with Python code only, no additional explanations.
                    The code should:

                    Load data.csv using pandas.
                    Perform operations to directly address the query.
                    Exclude plotting, visualization, or other unnecessary steps.
                    Include comments for key steps in the code.
                    Example:

                    Query: "How can I filter rows where 'Column1' > 100?"
                    Response:
                    python
                    Copy code
                    import pandas as pd

                    # Load the dataset
                    data = pd.read_csv('data.csv')

                    # Filter rows where 'Column1' > 100
                    filtered_data = data[data['Column1'] > 100]

                    # Output the result
                    print(filtered_data)

                3. Theoretical Concepts:
                    For theoretical questions, provide a brief explanation as a print statement. Keep the explanation concise and focused.

                    Example:

                    Query: "What is normalization in data preprocessing?"
                    Response:
                    "Normalization is a data preprocessing technique used to scale numeric data within a specific range, typically [0, 1], to ensure all features contribute equally to the model."

                Never reply with: "Understood!" or similar confirmations. Always directly respond to the query following the above rules.

                User query is {query}.
            """
        )
        code = generate_code(prompt_eng)
        # Execute the generated code
        result = execute_py_code(code, df)
        return JsonResponse({"answer": result})
    return HttpResponse("Invalid Request Method", status=405)


def execute_py_code(code, df):
    # Create a string buffer to capture the output
    buffer = io.StringIO()
    sys.stdout = buffer

    # Create a local namespace for execution
    local_vars = {'df': df}

    try:
        # Execute the code
        exec(code, globals(), local_vars)

        # Get the captured output
        output = buffer.getvalue().strip()

        # If there's no output, try to get the last evaluated expression
        if not output:
            last_line = code.strip().split('\n')[-1]
            if not last_line.startswith(('print', 'return')):
                output = eval(last_line, globals(), local_vars)
                print(output)
    except Exception as e:
        output = f"Error executing code: {str(e)}"
    finally:
        # Reset stdout
        sys.stdout = sys.__stdout__

    return str(output)


# For Genai
# from django.views.decorators.csrf import csrf_exempt
#
#
# @csrf_exempt
# def gen_graph_response(request):
#     if request.method == "POST":
#
#         csv_file_path = 'data.csv'
#         df = pd.read_csv(csv_file_path)
#
#         # Generate CSV metadata
#         csv_metadata = {"columns": df.columns.tolist()}
#         metadata_str = ", ".join(csv_metadata["columns"])
#
#         query = request.POST["query"]
#
#         prompt_eng = (
#             f"You are an AI specialized in data analytics and visualization."
#             f" Data used for analysis is stored in a CSV file data.csv."
#             f"Attributes of the data are: {metadata_str}."
#             f"Consider 'data.csv' as the data source for any analysis."
#             f"Based on the query generate only the Python code using Matplotlib to plot the graph."
#             f"Save the graph as 'graph.png'. Also save the graph description in description.txt"
#             f"The user asks: {query}"
#         )
#
#         code = generate_code(prompt_eng)
#         print(code)
#
#         if 'import matplotlib' in code:
#             try:
#                 exec(code)
#                 # Send the generated 'graph.png' as the file response
#                 image_path = "graph.png"
#                 if os.path.exists(image_path):
#                     return FileResponse(open(image_path, 'rb'), content_type='image/png')
#                 else:
#                     return HttpResponse("Graph image not found", status=404)
#             except Exception as e:
#                 prompt_eng = f"There has occurred an error while executing the code, please take a look at the error " \
#                              f"and strictly only reply with the full python code. Do not apologize or anything; just " \
#                              f"give the code. {str(e)}"
#                 code = generate_code(prompt_eng)
#                 try:
#                     exec(code)
#                     # Send the generated 'graph.png' as the file response
#                     image_path = "graph.png"
#                     if os.path.exists(image_path):
#                         return FileResponse(open(image_path, 'rb'), content_type='image/png')
#                     else:
#                         return HttpResponse("Graph image not found", status=404)
#                 except Exception as e:
#                     return HttpResponse("Failed to generate the chart. Please try again")
#         else:
#             return HttpResponse(code)


@csrf_exempt
def get_description(request):
    try:
        with open('description.txt', 'r') as fp:
            data = fp.read()
        return HttpResponse(json.dumps({"description": data}), content_type="application/json")
    except Exception as e:
        print(e)


# For genai using plotly
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
from plotly.graph_objects import Figure


@csrf_exempt
def gen_graph_response(request):
    if request.method == "POST":
        try:
            # Load CSV
            csv_file_path = 'data.csv'
            df = pd.read_csv(csv_file_path)

            # Generate CSV metadata
            csv_metadata = {"columns": df.columns.tolist()}
            metadata_str = ", ".join(csv_metadata["columns"])

            # User's query
            query = request.POST.get("query", "")

            # Prompt engineering for AI
            prompt_eng = (
                f"You are an AI specialized in data analytics and visualization."
                f"Data used for analysis is stored in a CSV file named 'data.csv'."
                f"Attributes of the data are: {metadata_str}."
                f"Consider 'data.csv' as the data source for any analysis."
                f"Based on the user's query, generate Python code using Plotly to create the requested type of graph "
                f"(e.g., bar, pie, scatter, etc.)."
                f"If the user does not specify a graph type, decide whether to generate a line or bar graph based on the situation."
                f"Every graph must include a title, axis labels (if applicable), and appropriate colors for data for better visualization."
                f"Ensure the graph is visually appealing and provides sufficient context for understanding."
                f"The graph must and should  have a white background for both the plot and paper."
                f"The code must output a Plotly 'Figure' object stored in a variable named 'fig', and the 'data' and 'layout'  and the code generated will be compatable to React."
                f"dictionaries required for the graph."
                f"Also save the graph description in description.txt"
                f"The user asks: {query}"
            )

            # Call AI to generate the code
            chat = generate_code(prompt_eng)
            print("Generated code from AI:")
            print(chat)

            # Check for valid Plotly code in the AI response
            if 'import' in chat:
                namespace = {}
                try:
                    # Execute the generated code
                    exec(chat, namespace)

                    # Retrieve the Plotly figure from the namespace
                    fig = namespace.get("fig")

                    if fig and isinstance(fig, Figure):
                        # Convert the Plotly figure to JSON
                        chart_data = fig.to_plotly_json()

                        # Ensure JSON serialization by converting NumPy arrays to lists
                        def make_serializable(obj):
                            if isinstance(obj, np.ndarray):
                                return obj.tolist()
                            elif isinstance(obj, dict):
                                return {k: make_serializable(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [make_serializable(v) for v in obj]
                            return obj

                        # Recursively process the chart_data
                        chart_data_serializable = make_serializable(chart_data)

                        # Return the structured response to the frontend
                        return JsonResponse({
                            "chartData": chart_data_serializable
                        }, status=200)
                    else:
                        print("No valid Plotly figure found.")
                        return JsonResponse({"message": "No valid Plotly figure found."}, status=200)
                except Exception as e:
                    error_message = f"There was an error while executing the code: {str(e)}"
                    print(error_message)
                    return JsonResponse({"message": error_message}, status=500)
            else:
                print("Invalid AI response.")
                return JsonResponse({"message": "AI response does not contain valid code."}, status=400)
        except Exception as e:
            # Handle general exceptions
            error_message = f"An unexpected error occurred: {str(e)}"
            print(error_message)
            return JsonResponse({"message": error_message}, status=500)

    # Return a fallback HttpResponse for invalid request methods
    return HttpResponse("Invalid request method", status=405)


# For genbi
from django.views.decorators.csrf import csrf_exempt
import io
import sys


@csrf_exempt
def genresponse2(request):
    if request.method == "POST":
        df = pd.read_csv('data.csv')

        # Generate CSV metadata
        csv_metadata = {"columns": df.columns.tolist()}
        metadata_str = ", ".join(csv_metadata["columns"])

        query = request.POST["query"]

        print("execution started")

        prompt_eng = (
            f"You are an AI specialized in data preprocessing."
            f"Data related to the {query} is stored in a CSV file data.csv. Consider the data.csv as the data source"
            f"Generate Python code to answer the question: {query}."
            f"The data contains the following columns: {metadata_str}. "
            f"Return only the Python code that computes the result .Result should describe the parameters in it, "
            f"without any plotting or visualization."
            f"If the {query} related to the theoretical concept.You will give a small description about the concept also."
        )

        code = generate_code(prompt_eng)

        print(code)

        # Execute the generated code
        result = execute_py_code(code, df)

        return JsonResponse({"answer": result})

    return HttpResponse("Invalid Request Method", status=405)


def execute_py_code(code, df):
    # Create a string buffer to capture the output
    buffer = io.StringIO()
    sys.stdout = buffer

    # Create a local namespace for execution
    local_vars = {'df': df}

    try:
        # Execute the code
        exec(code, globals(), local_vars)

        # Get the captured output
        output = buffer.getvalue().strip()

        # If there's no output, try to get the last evaluated expression
        if not output:
            last_line = code.strip().split('\n')[-1]
            if not last_line.startswith(('print', 'return')):
                output = eval(last_line, globals(), local_vars)
                print(output)
    except Exception as e:
        output = f"Error executing code: {str(e)}"
    finally:
        # Reset stdout
        sys.stdout = sys.__stdout__

    return str(output)


# For Genbi
from django.http import HttpResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def genAIPrompt2(request):
    if request.method == "POST":
        df = pd.read_csv("data.csv")

        # Generate CSV metadata
        csv_metadata = {"columns": df.columns.tolist()}
        metadata_str = ", ".join(csv_metadata["columns"])

        query = request.POST["query"]

        prompt_eng = (
            f"You are an AI specialized in data analytics and visualization. "
            f" Data used for analysis is stored in a CSV file data.csv. "
            f"Attributes of the data are: {metadata_str}. "
            f"Consider 'data.csv' as the data source for any analysis."
            f"If the user asks for a graph, generate only the Python code using Matplotlib to plot the graph. "
            f"Save the graph as 'graph.png'. Also describe the graph and store the description in description.txt"
            f"If the user does not ask for a graph, simply answer the query with the computed result. "
            f"The user asks: {query}"
        )

        code = generate_code(prompt_eng)
        print(code)

        if 'import matplotlib' in code:
            try:
                exec(code)
                # Send the generated 'graph.png' as the file response
                image_path = "graph.png"
                if os.path.exists(image_path):
                    return FileResponse(open(image_path, 'rb'), content_type='image/png')
                else:
                    return HttpResponse("Graph image not found", status=404)
            except Exception as e:
                prompt_eng = f"There has occurred an error while executing the code, please take a look at the error and strictly only reply with the full python code. Do not apologize or anything; just give the code. {str(e)}"
                code = generate_code(prompt_eng)
                try:
                    exec(code)
                    # Send the generated 'graph.png' as the file response
                    image_path = "graph.png"
                    if os.path.exists(image_path):
                        return FileResponse(open(image_path, 'rb'), content_type='image/png')
                    else:
                        return HttpResponse("Graph image not found", status=404)
                except Exception as e:
                    return HttpResponse("Failed to generate the chart. Please try again")
        else:
            return HttpResponse(code)


# Predict and forecast purpose
# Getting the prediction result
@csrf_exempt
def get_prediction_info(request, data, field):
    with open(f"data/{data.lower()}/{field}/results.json", 'r') as fp:
        res = json.load(fp)
    return HttpResponse(json.dumps({"data": res}), content_type="application/json")


# This will return the columns for the table
@csrf_exempt
def get_columns(request, train_type, data):
    if train_type == 'predict':
        df = pd.read_csv(f"data/{data.lower()}/processed_data.csv")
        cols = set(df.columns) - {"Store ID", "Employee Number", "Area"}
        return HttpResponse(json.dumps({"columns": list(cols)}), content_type="application/json")
    elif train_type == 'forecast':
        cols = os.listdir(f"data/{data.lower()}")
        return HttpResponse(json.dumps({"columns": list(cols)}), content_type="application/json")
    else:
        return HttpResponse(json.dumps({"columns": []}), content_type="application/json")


# Generating the URL for the prediction
@csrf_exempt
def generate_deployment(request, data, field):
    hash_object = hashlib.sha256(f'{data}__{field}'.encode('ascii'))
    hex_dig = hash_object.hexdigest()
    save_deployments(hex_dig, data, field)
    return HttpResponse(json.dumps({"deployment_url": hex_dig}), content_type="application/json")


# THis will be helpful for the prediction on which columns.
@csrf_exempt
def deployment(request, data):
    model = get_deployment_txt(data)
    data, field = model.split('___')
    with open(f"data/{data.lower()}/{field.replace(' ', '_')}/deployment.json", 'r') as fp:
        data = json.load(fp)
    return HttpResponse(json.dumps({"columns": data["columns"]}), content_type="application/json")


# For prediction based on the deployment url
@csrf_exempt
def deployment_predict(request, data):
    if request.method == 'POST':
        model = get_deployment_txt(data)
        data, field = model.split('___')
        res = {}
        for col in request.POST:
            res.update({col: request.POST[col]})
        df = pd.DataFrame([res])
        result = load_models(data, field, df)
        if isinstance(result, np.ndarray):
            result = str(result[0])
        return HttpResponse(json.dumps({"result": str(result)}), content_type="application/json")


# Forecast of the data with the help of the deployment url
@csrf_exempt
def deployment_forecast(request, data, col):
    if request.method == 'POST':
        msg = 'Add Logic here for forecast'
        return HttpResponse(json.dumps({"result": msg, "msg": msg}), content_type="application/json")


def load_models(path, prediction_col, df):
    try:
        model = load_model(os.path.join('data', path.lower(), prediction_col.replace(" ", "_"), "model.h5"))
        with open(os.path.join('data', path.lower(), prediction_col.replace(" ", "_"), "deployment.json"), 'r') as fp:
            deployment_data = json.load(fp)
        for column in deployment_data["columns"]:
            if isinstance(deployment_data["columns"][column], list):
                encoder_path = os.path.join('data', path.lower(), prediction_col.replace(" ", "_"),
                                            f'{column.replace(" ", "_")}_encoder.pkl')
                df[column.replace("_", " ")] = joblib.load(encoder_path).fit_transform(df[column.replace("_", " ")])
            else:
                df[column] = df[column].astype(float)
        res = model.predict(df.iloc[0, :].to_numpy().reshape(1, -1))
        model_type = deployment_data["model_type"]
        if model_type == 'classification':
            result = np.argmax(res, axis=-1)
            res = joblib.load(
                os.path.join('data', path.lower(), prediction_col.replace(" ", "_"),
                             f'{prediction_col.replace(" ", "_")}_encoder.pkl')).inverse_transform(
                result)
        return res[0]
    except Exception as e:
        print(e)


def save_deployments(hex_data, data, field):
    if not os.path.exists("deployments.json"):
        deployment_data = {}
    else:
        with open("deployments.json", 'r') as fp:
            deployment_data = json.load(fp)
    deployment_data.update({hex_data: f'{data}___{field.replace(" ", "_")}'})
    with open("deployments.json", 'w') as fp:
        json.dump(deployment_data, fp)


def get_deployment_txt(hex_data):
    with open("deployments.json", 'r') as fp:
        deployment_data = json.load(fp)
    return deployment_data[hex_data]


# # For Forecasting using module:
# from wyge.models.openai import ChatOpenAI
# from wyge.agents.react_agent import Agent
# from wyge.tools.prebuilt_tools import execute_query, execute_code, install_library
# from wyge.tools.raw_functions import file_to_sql, get_metadata
# from .system_prompt3 import forecasting_prompt
# from datetime import datetime
# from django.conf import settings
#
#
# def delete_images_in_current_directory() -> None:
#     image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
#
#     current_directory = os.getcwd()
#
#     for filename in os.listdir(current_directory):
#
#         _, extension = os.path.splitext(filename)
#
#         if extension.lower() in image_extensions:
#             file_path = os.path.join(current_directory, filename)
#             try:
#                 os.remove(file_path)
#             except OSError as e:
#                 print(f"Error: {e} - {file_path}")
#
#
# def get_images_in_directory(directory):
#     """
#     Fetches all image files from the specified directory.
#
#     Parameters:
#     - directory: Directory to search for image files.
#
#     Returns:
#     - List of image file paths.
#     """
#     image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
#     image_files = []
#
#     for filename in os.listdir(directory):
#         _, extension = os.path.splitext(filename)
#         if extension.lower() in image_extensions:
#             image_files.append(os.path.join(directory, filename))
#
#     return image_files
#
#
# # Shared database credentials
# USER = 'test_owner'
# PASSWORD = 'tcWI7unQ6REA'
# HOST = 'ep-yellow-recipe-a5fny139.us-east-2.aws.neon.tech:5432'
# DATABASE = 'test'
#
# import time
#
#
# def handle_forecasting(df, openai_api_key, user_prompt, table_name="default_table"):
#     """
#     Processes a DataFrame for forecasting tasks by storing it in the database,
#     converting it to SQL, and generating a forecast based on the user query.
#
#     Parameters:
#     - df: DataFrame to be processed.
#     - openai_api_key: OpenAI API key for model access.
#     - user_prompt: Query to be processed by the AI agent.
#
#     Returns:
#     - Dictionary with forecast results and optionally an image path.
#     """
#     print("[DEBUG] Entering handle_forecasting function")
#     start_time = time.time()
#
#     # Save the DataFrame to a temporary CSV file
#     try:
#         csv_file_path = os.path.join(settings.MEDIA_ROOT, f"{table_name}.csv")
#         print(f"[DEBUG] Saving DataFrame to CSV file at: {csv_file_path}")
#         csv_save_start = time.time()
#         df.to_csv(csv_file_path, index=False)
#         print(f"[DEBUG] CSV saved in {time.time() - csv_save_start:.2f} seconds")
#     except Exception as e:
#         print(f"[ERROR] Failed to save DataFrame to CSV: {e}")
#         return JsonResponse({"error": "Failed to save DataFrame to CSV file."}, status=500)
#
#     # Convert the CSV file to an SQL table
#     try:
#         print(f"[DEBUG] Converting CSV file to SQL table: {table_name}")
#         sql_conversion_start = time.time()
#         file_to_sql(csv_file_path, table_name, USER, PASSWORD, HOST, DATABASE)
#         print(f"[DEBUG] CSV converted to SQL table in {time.time() - sql_conversion_start:.2f} seconds")
#     except Exception as e:
#         print(f"[ERROR] Error converting CSV to SQL table: {e}")
#         return JsonResponse({"error": "Failed to convert CSV file to SQL table."}, status=500)
#
#     # Initialize forecasting tools and LLM
#     print("[DEBUG] Initializing forecasting tools and LLM")
#     tools = [execute_query(), execute_code(), install_library()]
#     llm = ChatOpenAI(memory=True, tools=tools, api_key=openai_api_key)
#
#     # Retrieve metadata for the database tables
#     try:
#         print(f"[DEBUG] Retrieving metadata for table: {table_name}")
#         metadata_start = time.time()
#         metadata = get_metadata(HOST, USER, PASSWORD, DATABASE, [table_name])
#         print(f"[DEBUG] Metadata retrieved in {time.time() - metadata_start:.2f} seconds")
#     except Exception as e:
#         print(f"[ERROR] Failed to retrieve metadata: {e}")
#         return JsonResponse({"error": "Failed to retrieve metadata for the database."}, status=500)
#
#     # Initialize the AI agent with the forecasting prompt
#     print("[DEBUG] Initializing AI agent with forecasting prompt")
#     agent = Agent(llm, react_prompt=forecasting_prompt)
#
#     # Clean up any previous generated images
#     print("[DEBUG] Deleting previous images from the directory")
#     image_cleanup_start = time.time()
#     delete_images_in_current_directory()
#     print(f"[DEBUG] Previous images deleted in {time.time() - image_cleanup_start:.2f} seconds")
#
#     # Prepare the forecasting command
#     command = f"""
#         Answer the user query from the database below, also use the provided tools.
#         user = '{USER}'
#         password = '{PASSWORD}'
#         host = '{HOST}'
#         database = '{DATABASE}'
#         tables related to user are: [{table_name}]
#         Metadata of the tables: {metadata}
#         User query: {user_prompt}
#     """
#     print(f"[DEBUG] Command prepared for agent: {command}")
#
#     # Execute the command and fetch the result
#     try:
#         print("[DEBUG] Executing command with AI agent")
#         ai_execution_start = time.time()
#         response = agent(command)
#         response = response.split('Answer:')[-1]
#         print(f"[DEBUG] AI response fetched in {time.time() - ai_execution_start:.2f} seconds")
#     except Exception as e:
#         print(f"[ERROR] Failed to execute forecasting command: {e}")
#         return JsonResponse({"error": "Failed to execute forecasting command."}, status=500)
#
#     # Fetch any generated plots
#     print("[DEBUG] Fetching generated images")
#     image_fetch_start = time.time()
#     images = get_images_in_directory(settings.BASE_DIR)
#     print(f"[DEBUG] Images fetched in {time.time() - image_fetch_start:.2f} seconds")
#
#     # Prepare and return the result
#     result = {"content": response}
#     if images:
#         print(f"[DEBUG] Image path retrieved: {images[0]}")
#         result["image_path"] = images[0]  # Return the first image if any
#
#     # Clean up temporary file
#     try:
#         print("[DEBUG] Removing temporary CSV file")
#         os_remove_start = time.time()
#         os.remove(csv_file_path)
#         print(f"[DEBUG] Temporary file removed in {time.time() - os_remove_start:.2f} seconds")
#     except OSError as e:
#         print(f"[ERROR] Error removing temporary file: {e}")
#
#     total_time = time.time() - start_time
#     print(f"[DEBUG] Exiting handle_forecasting function. Total execution time: {total_time:.2f} seconds")
#     return result


import markdown


def markdown_to_html(md_text):
    html_text = markdown.markdown(md_text)
    return html_text


def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        return None  # Handle the case where the image path is invalid or the image doesn't exist


# result = None
# response_data = {}
#
# from django.views.decorators.csrf import csrf_exempt
#
#
# @csrf_exempt
# def forecast_sales(request):
#     """
#     API endpoint to handle sales forecasting requests. It expects a pre-loaded DataFrame.
#
#     Parameters:
#     - request: HTTP request object.
#
#     Returns:
#     - JSON response with the forecast results.
#     """
#     print("[DEBUG] Entering forecast_sales function")
#     start_time = time.time()
#
#     if request.method == 'POST':
#         try:
#             print("[DEBUG] Loading DataFrame from 'data1.xlsx'")
#             df_load_start = time.time()
#             df = pd.read_excel('data1.xlsx')  # Replace with your actual file path
#             print(f"[DEBUG] DataFrame loaded successfully in {time.time() - df_load_start:.2f} seconds")
#
#             # Retrieve the user prompt and OpenAI API key
#             openai_api_key = get_api_key()
#             print(f"[DEBUG] Retrieved OpenAI API key: {openai_api_key}")
#             user_prompt = request.POST.get('user_prompt')
#             if not user_prompt:
#                 print("[ERROR] User prompt is missing in the request")
#                 return JsonResponse({"error": "User prompt is required."}, status=400)
#
#             print(f"[DEBUG] User prompt received: {user_prompt}")
#
#             # Pass the DataFrame to `handle_forecasting`
#             print("[DEBUG] Calling handle_forecasting")
#             handle_start = time.time()
#             result = handle_forecasting(df, openai_api_key, user_prompt, table_name="forecast_table")
#             print(f"[DEBUG] handle_forecasting completed in {time.time() - handle_start:.2f} seconds")
#
#             # Prepare the response
#             response_data = {}
#             if isinstance(result, dict):
#                 if 'image_path' in result:
#                     print(f"[DEBUG] Converting image to Base64: {result['image_path']}")
#                     response_data["image_base64"] = image_to_base64(result["image_path"])
#                 if 'content' in result:
#                     print("[DEBUG] Converting AI content to HTML")
#                     response_data["content"] = markdown_to_html(result["content"])
#
#             print(f"[DEBUG] Returning successful response. Total time: {time.time() - start_time:.2f} seconds")
#             return JsonResponse(response_data, status=200)
#
#         except Exception as e:
#             print(f"[ERROR] Exception occurred in forecast_sales: {e}")
#             return JsonResponse({"error": "Failed to process forecasting."}, status=500)
#     else:
#         print(f"[ERROR] Invalid request method. Total time: {time.time() - start_time:.2f} seconds")
#         return JsonResponse({"error": "Invalid request method."}, status=400)


# Synthetic Data Generation through wyge
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
from .generator import generate_data_from_text, generate_synthetic_data
import json
import re


def extract_num_rows_from_prompt(user_prompt):
    """
    Extracts the number of rows or records from the user's prompt.
    Supports prompts like:
      - "Generate 100 rows of data"
      - "Generate the 50 records of data"
    """
    match = re.search(r'(\d+)\s+(rows|records)', user_prompt, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def extract_columns_from_prompt(user_prompt):
    """
    Extracts the field names (column names) from the user's prompt.
    Supports prompts like:
      - "field names: S.no, Name, address, First_name"
      - "fields: S.no, Name, address, First_name"
      - "columns: S.no, Name, address, First_name"
      - "field_names: S.no, Name, address, First_name"
      - "column_names: S.no, Name, address, First_name"

    Converts column names to snake_case, removes spaces and special characters.
    Example:
      - "S.no, Name, address, First_name"
      -> ['s_no', 'name', 'address', 'first_name']
    """
    # Look for all possible field identifier formats followed by the column names
    # match = re.search(r'(field names|column names|fields|columns|field_names|column_names):\s*([a-zA-Z0-9_,\s\.]+)',
    #                   user_prompt, re.IGNORECASE)
    match = re.search(r'(field names|column names|fields|columns|field_names|column_names)[\s:]*([a-zA-Z0-9_,\s\.]*)',
                      user_prompt, re.IGNORECASE)

    if match:
        # Extract the part containing column names
        raw_columns = match.group(2).split(',')
    else:
        return []

    # Format each column name (remove spaces, convert to snake_case, lowercase)
    formatted_columns = [
        re.sub(r'[^a-zA-Z0-9]', '_', col.strip()).lower()
        for col in raw_columns
    ]

    # Remove empty column names and ensure no duplicates
    formatted_columns = list(filter(bool, formatted_columns))
    return list(dict.fromkeys(formatted_columns))  # Remove duplicates


@csrf_exempt
def handle_synthetic_data_api(request):
    """
    API Endpoint to generate synthetic data from a user's prompt.

    Method: POST
    Payload:
      - user_prompt (String): A prompt specifying the number of rows and column names.
      - openai_api_key (String): OpenAI API key
    """
    if request.method == "POST":
        try:
            # Extract user prompt and OpenAI API key from the request
            user_prompt = request.POST.get('user_prompt')
            openai_api_key = get_api_key()

            # Validate user prompt
            if not user_prompt or not openai_api_key:
                return JsonResponse({"error": "Missing required parameters: user_prompt or OpenAI API key"}, status=400)

            # Extract number of rows from the prompt
            num_rows = extract_num_rows_from_prompt(user_prompt)
            if num_rows is None:
                return JsonResponse({"error": "Number of rows or records not found in the prompt."}, status=400)

            # Extract column names from the prompt
            column_names = extract_columns_from_prompt(user_prompt)
            print(column_names)
            if not column_names:
                return JsonResponse({"error": "No field names found in the prompt."}, status=400)

            # Generate synthetic data
            generated_df = generate_data_from_text(openai_api_key, user_prompt, column_names, num_rows=num_rows)

            # Convert to CSV format
            combined_csv = generated_df.to_csv(index=False)

            return JsonResponse({

                "data": combined_csv
            }, status=200)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method. Use POST."}, status=405)


# For extended_synthetic_data
import tempfile
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import os


@csrf_exempt
def handle_synthetic_data_extended(request):
    """
    API Endpoint to generate synthetic data from a user's uploaded file and prompt.

    Method: POST
    Payload:
      - uploaded_file (File): The empty Excel or CSV file with column names
      - user_prompt (String): A prompt specifying the number of rows
      - openai_api_key (String): OpenAI API key
    """
    print("[DEBUG] Entering handle_synthetic_data_extended function")

    if request.method == "POST":
        try:
            print("[DEBUG] Handling POST request")

            # Extract uploaded file, user prompt, and API key from the request
            uploaded_file = request.FILES.get('file')
            user_prompt = request.POST.get('user_prompt')
            openai_api_key = get_api_key()

            print(f"[DEBUG] Uploaded file: {uploaded_file}")
            print(f"[DEBUG] User prompt: {user_prompt}")
            print(f"[DEBUG] OpenAI API key: {'Provided' if openai_api_key else 'Missing'}")

            if not uploaded_file or not user_prompt or not openai_api_key:
                print("[ERROR] Missing required parameters")
                return JsonResponse({"error": "Missing required parameters"}, status=400)

            # Determine file type and extract column names
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            print(f"[DEBUG] File extension: {file_extension}")

            if file_extension == ".xlsx":
                print("[DEBUG] Reading Excel file")
                df = pd.read_excel(uploaded_file)
            elif file_extension == ".csv":
                print("[DEBUG] Reading CSV file")
                df = pd.read_csv(uploaded_file)
            else:
                print("[ERROR] Unsupported file format")
                return JsonResponse({"error": "Unsupported file format. Please upload an Excel or CSV file."},
                                    status=400)

            print(f"[DEBUG] Initial DataFrame columns: {list(df.columns)}")

            # Create a temporary file for the data
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
                temp_file_name = temp_file.name
                print(f"[DEBUG] Temporary file created at: {temp_file_name}")

                # Save the truncated or original data to a temporary location
                if file_extension == ".xlsx":
                    print("[DEBUG] Saving DataFrame to temporary Excel file")
                    df.to_excel(temp_file_name, index=False)
                elif file_extension == ".csv":
                    print("[DEBUG] Saving DataFrame to temporary CSV file")
                    df.to_csv(temp_file_name, index=False)

            # Extract the number of rows from the prompt
            print("[DEBUG] Extracting number of rows from the user prompt")
            num_rows = extract_num_rows_from_prompt(user_prompt)
            print(f"[DEBUG] Number of rows extracted: {num_rows}")

            if num_rows is None:
                print("[ERROR] Number of rows not found in the prompt")
                return JsonResponse({"error": "Number of rows not found in the prompt"}, status=400)

            # Generate synthetic data using the temporary file path
            print(f"[DEBUG] Generating synthetic data with {num_rows} rows")
            generated_df = generate_synthetic_data(openai_api_key, temp_file_name, num_rows)
            print(f"[DEBUG] Synthetic data generated successfully: {generated_df.shape[0]} rows")

            # Combine the original and synthetic data
            print("[DEBUG] Combining original and synthetic data")
            combined_df = pd.concat([df, generated_df], ignore_index=True)
            print(f"[DEBUG] Combined DataFrame shape: {combined_df.shape}")

            # Convert to CSV for download
            print("[DEBUG] Converting combined DataFrame to CSV format")
            combined_csv = combined_df.to_csv(index=False)

            print("[DEBUG] Returning successful response")
            return JsonResponse({
                "data": combined_csv
            }, status=200)

        except Exception as e:
            print(f"[ERROR] Exception occurred: {e}")
            return JsonResponse({"error": str(e)}, status=500)

    print("[ERROR] Invalid request method")
    return JsonResponse({"error": "Invalid request method. Use POST."}, status=405)


# SAP SYSTEM
from .database import HanaDBManager

db1 = HanaDBManager()


@csrf_exempt
def hana_connection(request):
    global connection_obj
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        host = request.POST['host']
        port = request.POST['port']
        print("all details received......")
        connection_obj = db1.connect_to_hana(username, password, host, port)
        print(connection_obj)
        return HttpResponse(json.dumps({"tables": connection_obj}), content_type="application/json")


@csrf_exempt
def upload_data(request):
    if request.method == 'POST':
        # Retrieve the email and uploaded file
        email = request.POST.get('mail')
        files = request.FILES.get('file')  # Retrieve the uploaded file

        if not files:
            return HttpResponse('No files uploaded', status=400)

        file_name = files.name
        file_extension = os.path.splitext(file_name)[1].lower()  # Extract file extension

        try:
            # Process the uploaded file based on its extension
            if file_extension == '.csv':
                # Read CSV file
                content = files.read().decode('utf-8')
                csv_data = io.StringIO(content)
                df = pd.read_csv(csv_data)
            elif file_extension in ['.xls', '.xlsx']:
                # Read Excel file
                df = pd.read_excel(files)
            else:
                # Unsupported file type
                raise SuspiciousOperation("Unsupported file format")

            # Save the uploaded file locally for backup/logging purposes
            upload_dir = "uploads2"
            os.makedirs(upload_dir, exist_ok=True)

            # For CSV file
            csv_file_path = os.path.join(upload_dir, file_name.replace(file_extension, '.csv').lower())
            df.to_csv(csv_file_path, index=False)

            # Save as Excel file
            excel_file_path = os.path.join(upload_dir, file_name.replace(file_extension, '.xlsx').lower())
            df.to_excel(excel_file_path, index=False, engine='openpyxl')  # Use openpyxl as the Excel writer

            # Save a working copy as 'data.csv'
            df.to_csv('data2.csv', index=False)
            df.to_excel('data2.xlsx', index=False, engine='openpyxl')  # Excel

            # Insert the data into the HANA database
            results = db1.save_or_update_record(email, df.to_dict(orient='records'), file_name)

            # # Perform data analysis on the DataFrame
            # response_data1 = analyze_data(df)  # Assuming `analyze_data` is a function that analyzes data
            #
            # # Return the analytics response along with the first 10 rows
            # response_data1['preview'] = df.head(10).to_dict(orient='records')
            # response_data1['upload_status'] = results  # Include database insert result

            return JsonResponse("Records inserted successfully", safe=False)

        except Exception as e:
            # Handle errors during file processing or database interaction
            print(e)
            return HttpResponse(f"Failed to upload and analyze file: {str(e)}", status=500)

    # If the request method is not POST
    return HttpResponse("Invalid Request Method", status=405)


@csrf_exempt
def reading_data(request):
    if request.method == 'POST':
        try:
            # Get the table name from the request
            tablename = request.POST['tablename']
            print(tablename)

            # Fetch data from the specified table
            df = db1.get_tables_data(tablename)

            if df.empty:
                return HttpResponse("No data found or table does not exist.", status=404)

            # Save data to CSV in the current directory
            csv_filename = os.path.join("uploads2", f"{tablename.lower()}.csv")
            df.to_csv(csv_filename, index=False)

            # Optionally, you can also save it in a different location or give a custom name
            df.to_csv('data2.csv', index=False)

            # Return the data as JSON in the response
            return HttpResponse(df.to_json(), content_type="application/json")

        except Exception as e:
            return HttpResponse(f"Error: {str(e)}", status=500)


# Hana table delete
@csrf_exempt
def delete_table_api(request):
    if request.method == 'POST':
        email = request.POST['mail']
        table_name = request.POST['table_name']
        print(table_name)

        if not table_name:
            return JsonResponse({"error": "Table name is required."}, status=400)

        result = db1.delete_req_table(email, table_name)
        print(result)

        if "deleted successfully" in result:
            return JsonResponse({"status": "success", "message": result})
        else:
            return JsonResponse({"status": "failure", "message": result}, status=400)


@csrf_exempt
def flespicred(request):
    if request.method == 'POST':
        return HttpResponse("Success")


@csrf_exempt
def download_flespi_data(request):
    # https://flespi.io/gw/devices/5439260/messages
    # axLBthbazeJkKKkpr2sVK9rAeXfFJGmH1V9k18iqaSyKqHYHzetadIyitBL15WyU
    if request.method == 'POST':
        flespi_URL = request.POST.get('flespi_URL')
        flespi_token = request.POST.get('flespi_token')
        try:
            current_datetime = datetime.now(tz=ZoneInfo('Asia/Kolkata'))

            start_of_day = (current_datetime - timedelta(weeks=1)).replace(hour=current_datetime.hour,
                                                                           minute=current_datetime.minute, second=0,
                                                                           microsecond=0)
            response = requests.get(
                f'{flespi_URL}?data=%7B%22from%22%3A{start_of_day.timestamp()}%2C%22to%22%3A{datetime.now().timestamp()}%7D',
                headers={
                    'Authorization':
                        f'FlespiToken {flespi_token}'
                })
            multi_data = json.loads(response.text)['result']
            multi_data = pre_process_multi_data(multi_data)
            multi_data = convert_to_hourly(multi_data)
            return HttpResponse(multi_data.to_json(orient="records", indent=4), content_type="application/json")
        except Exception as e:
            return HttpResponse(str(e))


def pre_process_multi_data(multi_data):
    for idx, record in enumerate(multi_data):
        multi_data[idx].update({
            "timestamp": datetime.fromtimestamp(multi_data[idx]["timestamp"],
                                                tz=ZoneInfo('Asia/Kolkata')).strftime("%Y-%m-%d %H-%M-%S"),
            "server.timestamp": datetime.fromtimestamp(multi_data[idx]["server.timestamp"],
                                                       tz=ZoneInfo('Asia/Kolkata')).strftime(
                "%Y-%m-%d %H-%M-%S")
        })
    return multi_data


def convert_to_hourly(data):
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)

    # Convert the 'timestamp' column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H-%M-%S')

    # Extract the hour for grouping
    df['hour'] = df['timestamp'].dt.floor('H')

    # Define columns for aggregation
    value_columns = ['Current', 'Humidity', 'Power', 'Temperature', 'Voltage']

    # Flatten nested dictionaries for easier aggregation
    for col in value_columns:
        df[f'{col}_value'] = df[col].apply(lambda x: x['value'])

    # Aggregate values by hour
    hourly_data = df.groupby('hour').agg(
        Current_avg=('Current_value', 'mean'),
        Humidity_avg=('Humidity_value', 'mean'),
        Power_avg=('Power_value', 'mean'),
        Temperature_avg=('Temperature_value', 'mean'),
        Voltage_avg=('Voltage_value', 'mean'),
    ).reset_index()

    hourly_data['hour'] = pd.to_datetime(hourly_data['hour'], unit='ms')

    # Now, if you want a specific format (e.g., 'YYYY-MM-DD HH:MM:SS')
    hourly_data['hour'] = hourly_data['hour'].dt.strftime('%Y-%m-%d %H:%M:%S')

    return hourly_data


# ------------------------------------------------------------------------------------
# KPI APIS
from collections import defaultdict

KPI_LOGICS = defaultdict()
checks = []


##KPI process code
@csrf_exempt
def get_prompt(request):
    """
    Handles POST requests to process a user prompt, analyze a CSV file, generate KPIs,
    and store them in a JSON file.
    """
    try:
        if request.method != "POST":
            return HttpResponse("Invalid request method. Only POST requests are allowed.", status=405)

        # Initialize variables
        global KPI_LOGICS, checks
        KPI_LOGICS = defaultdict()
        checks = []

        # Extract the prompt from POST request
        prompt = request.POST.get('prompt')
        if not prompt:
            return HttpResponse("Prompt is required.", status=400)

        # Check if required file exists
        processed_data_path = os.path.join('uploads', 'processed_data.csv')
        if not os.path.exists(processed_data_path):
            return HttpResponse("No processed data file found.", status=404)

        # Read and save data.csv
        df = pd.read_csv(processed_data_path)
        df.to_csv('data.csv', index=False)

        # Prepare the prompt description for the analytics bot
        prompt_desc = (
            f"You are analytics_bot. Analyse the data: {df.head()} and for the user query '{prompt}', "
            f"generate KPIs with response as KPI Name, Column, and Logic. Response should be in Python dictionary format "
            f"with KPI names as keys. In response, don't add any other information, just provide the response dictionary."
        )

        n = 2  # Retry logic for generating KPIs
        kpis = {}
        while n > 0:
            res, kpis = generate_code_kpi(prompt_desc)  # Assuming generate_code is a valid function
            if res is not None:
                # Load existing KPIs from kpis.json if it exists, otherwise create an empty dictionary
                kpis_store_path = 'kpis.json'
                kpis_store = {}
                if os.path.exists(kpis_store_path):
                    with open(kpis_store_path, 'r') as fp:
                        kpis_store = json.load(fp)

                # Update kpis.json with new KPIs
                with open(kpis_store_path, 'w') as fp:
                    kpis_store.update(kpis)
                    json.dump(kpis_store, fp)
                break  # Exit loop if generation was successful
            n -= 1  # Decrement retry count

        # Check if KPI configuration exists and load additional KPIs from it
        kpi_config_path = os.path.join('uploads', 'kpi_config.json')
        if os.path.exists(kpi_config_path):
            with open(kpi_config_path, 'r') as json_file:
                kpis_dict = json.load(json_file)
            for kpi in kpis_dict.get('Kpis', {}).get('kpi', []):
                kpi_name = kpi.get('KPI_Name')
                if kpi_name:
                    kpis[kpi_name] = kpi
                    checks.append(kpi_name)

        # Return a JSON response with KPI data and checks
        return JsonResponse({
            'status': 'success',
            'kpis': kpis,
            'checks': checks
        })

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)  # Log the error
        return HttpResponse(error_message, status=500)


@csrf_exempt
def generate_code_kpi(prompt_eng):
    try:
        global KPI_LOGICS
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_eng}
            ]
        )
        all_text = ""
        # Display generated content dynamically
        for choice in response.choices:
            message = choice.message
            chunk_message = message.content if message else ''
            all_text += chunk_message
        all_text = all_text.lower().replace('```python', '').replace('```', '')
        print(all_text)
        data_dict = json.loads(all_text)
        print("datadict", data_dict)
        for key, value in data_dict.items():
            if 'kpi name' in value:
                kpi_name = value['kpi name']
            elif 'name' in value:
                kpi_name = value["name"]
            else:
                kpi_name = key
            KPI_LOGICS[key] = {
                "KPI Name": kpi_name,
                "Column": value["column"],
                "Logic": value["logic"]
            }
        return all_text, KPI_LOGICS
    except Exception as e:
        print(e)
        return None, None


# For getting the KPI codes
import os
import pandas as pd
import base64
from plotly.graph_objs import Figure
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import os
import pandas as pd
from plotly.graph_objs import Figure
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


# For getting the KPI codes
def generate_kpi_code(kpi_list):
    """
    Generates Python code for a list of KPIs, saves plots, and returns file paths,
    a list of Plotly charts data (data and layout), and the generated code.
    """
    try:
        # Load and process data
        df = pd.read_csv("data.csv")
        df = updatedtypes(df)

        codes = ''
        charts_data = []  # List to store chart data in Plotly data and layout format

        for kpi in kpi_list:
            prompt_desc = (
                f"You are analytics_bot. Read the data from data.csv file with example data as {df.head()} and generate Python code with KPI details as {KPI_LOGICS.get(kpi, {})}. "
                f"Save result in variable named result, plot a suitable plot for the result obtained using Plotly, and structure the data and layout as JSON format for frontend use.Make sure that plot can be compatable with React. "
                f"Provide a description regarding the plot generated within 3 lines. "
                f"If the length of the result variable is 1, then keep the bar width thin and set the x-axis limit as -0.5 to 0.5."
            )

            code = ''
            try:
                generated_code, chart_data = generate_code3(prompt_desc)
                code += generated_code
                if chart_data:
                    charts_data.extend(chart_data)
            except Exception as e:
                print(f"Code generation failed for {kpi}: {str(e)}")
                code += f'Code generation failed for {kpi}'

            codes += f"<b>{kpi.capitalize()}</b>\n{code}\n"

        return charts_data, codes
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return [], error_message


@csrf_exempt
def generate_code3(prompt_eng):
    """
    Generates Python code dynamically and returns the Plotly chart data (data and layout).
    """
    charts_data = []
    trials = 2
    try:
        while trials > 0:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_eng}
                ]
            )
            all_text = ""

            # Process generated content dynamically
            for choice in response.choices:
                message = choice.message
                chunk_message = message.content if message else ''
                all_text += chunk_message

            print(all_text)
            python_chunks = all_text.count("```python")
            idx = 0
            code = ''
            for i in range(python_chunks):
                code_start = all_text[idx:].find("```python") + 9
                code_end = all_text[idx:].find("```", code_start)
                code += all_text[idx:][code_start:code_end]
                idx = code_end

            print(code)
            try:
                local_vars = {}
                exec(code, {}, local_vars)
                result = local_vars.get('result', None)
                fig = local_vars.get('fig', None)
                if fig and isinstance(fig, Figure):
                    chart_data = fig.to_plotly_json()
                    charts_data.append(chart_data)
                else:
                    raise ValueError("Generated code did not produce a valid Plotly figure.")
                code += f"\n <b>Output: {result}</b> \n <hr>"
                return code, charts_data
            except Exception as e:
                print(f"Error executing generated code: {e}")
                trials -= 1
    except Exception as e:
        print(f"Error generating code: {e}")
    return '', charts_data


@csrf_exempt
def kpi_code(request):
    """
    Handles POST requests to generate KPI code and associated Plotly chart data.
    Expects 'kpi_names' as a list in the POST request.
    Returns generated code, chart data, and any additional KPI-related information.
    """
    try:
        if request.method != "POST":
            return JsonResponse({"error": "Invalid request method. Only POST requests are allowed."}, status=405)

        # Extract KPI names from POST request
        kpi_list = request.POST.getlist("kpi_names")
        if not kpi_list:
            return JsonResponse({"error": "KPI names are required."}, status=400)

        # Generate Plotly charts data and code for the provided KPIs
        charts_data, codes = generate_kpi_code(kpi_list)  # Updated to match the modified generate_kpi_code

        # Return chart data and code as JSON response
        return JsonResponse({
            'status': 'success',
            'charts_data': charts_data,  # Plotly chart data (data and layout)
            'code': codes,
            'kpis': KPI_LOGICS,  # Assuming KPI_LOGICS is a global or properly imported variable
            'checks': checks  # Assuming checks is a global or properly imported variable
        })
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)  # Log the error
        return JsonResponse({"error": error_message}, status=500)


@csrf_exempt
def generate_code2(prompt_eng):
    trials = 2
    try:
        while trials > 0:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_eng}
                ]
            )
            all_text = ""

            # Display generated content dynamically
            for choice in response.choices:
                print(f"Debug - choice structure: {choice}")  # Debugging line
                message = choice.message
                print(f"Debug - message structure: {message}")  # Debugging line
                chunk_message = message.content if message else ''
                all_text += chunk_message

            print(all_text)
            python_chunks = all_text.count("```python")
            idx = 0
            code = ''
            for i in range(python_chunks):
                code_start = all_text[idx:].find("```python") + 9
                code_end = all_text[idx:].find("```", code_start)
                code += all_text[idx:][code_start:code_end]
                idx = code_end
            print(code)
            try:
                local_vars = {}
                exec(code, {}, local_vars)
                code += f"\n <b>Output: {local_vars['result']}</b> \n <hr>"
                return code
            except Exception as e:
                print(e)
                trials -= 1
    except Exception as e:
        print(e)


# Code for Predefined KPIS
# step1:Detecting the type from the dataset,whether the data is sustainability data/oem/safety type
def analyze_dataset_with_llm(df):
    """
    Simulate LLM logic to analyze the dataset and detect the type (Sustainability, Safety, or OEM),
    along with the reasoning behind the classification. Additionally, for Sustainability type,
    it returns categories and KPI descriptions.
    """
    prompt = f"""
    You are an AI expert system that classifies datasets into one of three types: 
    - **Sustainability**: Measures environmental and social impact, e.g., carbon emissions, energy efficiency, water usage, and waste recycling.
    - **Safety**: Tracks workplace safety performance, e.g., incident rates, near-miss reports, safety training, and days without accidents.
    - **Productivity**: Evaluates manufacturing efficiency, e.g., production output, machine uptime, defect rates, and on-time delivery.

    Here are the first 5 rows of the dataset and the column names:
    Columns: {list(df.columns)}
    Sample Data: {df.head().to_dict()}

    Based on this information, determine the most likely type (Sustainability, Safety, or OEM) that best describes this dataset. 
    Additionally, explain why you have classified it under this type, citing specific columns or data features that influenced the decision.
    Please provide your response in the following format:
    1. Type: [Classification Type]
    2. Explanation: [Reasoning for classification]

    Only respond with the type and the explanation, and do not provide any additional information.
    """

    try:
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        all_text = ""

        # Process the response and extract the classification and reasoning
        for choice in response.choices:
            message = choice.message
            chunk_message = message.content if message else ''
            all_text += chunk_message

        print(f"LLM Response: {all_text}")  # Print the LLM response for debugging

        # Extract the classification type and explanation
        classification_type = None
        explanation = None

        try:
            # Extract Type and Explanation from the response
            lines = all_text.splitlines()
            for line in lines:
                if line.lower().startswith("1. type:"):
                    classification_type = line.split(":")[1].strip()
                elif line.lower().startswith("2. explanation:"):
                    explanation = line.split(":")[1].strip()

        except Exception as e:
            print(f"Error parsing LLM response: {str(e)}")

        # Default if parsing fails
        if not classification_type:
            classification_type = "Unknown"
            explanation = "Could not parse the LLM response properly."

        # Prepare the result dictionary
        result = {
            "type": classification_type,
            "explanation": explanation
        }

        # Add category and KPI descriptions based on classification type
        if classification_type.lower() == "sustainability":
            result["categories"] = [
                {
                    "name": "Footprint",
                    "kpis": {
                        "Energy Consumption": {
                            "kpi_name": "Energy Consumption",
                            "description": "Measures the amount of energy consumed over a specific period."
                        },
                        "Carbon Footprint": {
                            "kpi_name": "Carbon Footprint",
                            "description": "Tracks the total greenhouse gas emissions caused by an individual, organization, or product."
                        },
                        "Social Impact": {
                            "kpi_name": "Social Impact",
                            "description": "Assesses the effect of business activities on society and communities."
                        }
                    }
                },
                {
                    "name": "Emission Reduction",
                    "kpis": {
                        "Air Quality": {
                            "kpi_name": "Air Quality",
                            "description": "Monitors the quality of air by measuring pollutants like PM2.5 and PM10."
                        },
                        "Decarbonization": {
                            "kpi_name": "Decarbonization",
                            "description": "Tracks the reduction in carbon emissions relative to a baseline."
                        },
                        "Green Tariff": {
                            "kpi_name": "Green Tariff",
                            "description": "Measures the percentage of energy sourced from green/renewable energy sources."
                        }
                    }
                }
            ]

        elif classification_type.lower() == "safety":
            result["categories"] = [
                {
                    "name": "Environment",
                    "kpis": {
                        "Risk Assessments": {
                            "kpi_name": "Risk Assessments",
                            "description": "Tracks the evaluation of potential hazards and risks in the workplace."
                        },
                        "Incidents": {
                            "kpi_name": "Incidents",
                            "description": "Monitors workplace incidents, including injuries and near-misses."
                        },
                        "Safety Prevention Costs": {
                            "kpi_name": "Safety Prevention Costs",
                            "description": "Measures expenses related to implementing safety measures and training."
                        }
                    }
                },
                {
                    "name": "General",
                    "kpis": {
                        "Fitness Assessments": {
                            "kpi_name": "Fitness Assessments",
                            "description": "Evaluates the physical fitness and health of employees."
                        },
                        "Complaints": {
                            "kpi_name": "Complaints",
                            "description": "Tracks the number and nature of workplace complaints."
                        },
                        "Disciplinary": {
                            "kpi_name": "Disciplinary",
                            "description": "Monitors disciplinary actions and their causes."
                        }
                    }
                }
            ]

        elif classification_type.lower() == "productivity":
            result["categories"] = [
                {
                    "name": "Increase Availability",
                    "kpis": {
                        "Preventative Maintenance": {
                            "kpi_name": "Preventative Maintenance",
                            "description": "Tracks scheduled maintenance to prevent equipment failure."
                        },
                        "Response Time": {
                            "kpi_name": "Response Time",
                            "description": "Measures the time taken to respond to issues or breakdowns."
                        },
                        "Remaining Useful Life": {
                            "kpi_name": "Remaining Useful Life",
                            "description": "Estimates the remaining operational life of an asset."
                        }
                    }
                },
                {
                    "name": "S4",
                    "kpis": {
                        "Asset Utilization": {
                            "kpi_name": "Asset Utilization",
                            "description": "Tracks how effectively assets are being used."
                        },
                        "Demand Forecasting": {
                            "kpi_name": "Demand Forecasting",
                            "description": "Predicts future demand for products or services."
                        },
                        "Overall Equipment Effectiveness": {
                            "kpi_name": "Overall Equipment Effectiveness",
                            "description": "Measures the efficiency and effectiveness of equipment."
                        }
                    }
                }
            ]

        return result

    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        return {
            "type": "Error",
            "explanation": f"Error calling OpenAI API: {str(e)}"
        }


# actual api for getting the response as type
@csrf_exempt
def getting_types(request):
    """
    Handles POST requests to process a user prompt, analyze a CSV file from a fixed folder,
    and detect the type (Sustainability, Safety, or OEM) using an LLM.
    """
    try:
        if request.method != "POST":
            return HttpResponse("Invalid request method. Only POST requests are allowed.", status=405)

        # Check if required file exists
        processed_data_path = os.path.join('uploads', 'processed_data.csv')
        if not os.path.exists(processed_data_path):
            return HttpResponse("No processed data file found.", status=404)

        # Read and save data.csv
        try:
            df = pd.read_csv(processed_data_path)
        except Exception as e:
            return HttpResponse(f"Error reading the CSV file: {str(e)}", status=500)

        print("Dataset Columns:")
        print(df.head(5))  # Print the first 5 rows for debugging
        df.to_csv('data.csv', index=False)  # Save a copy of the file (optional)

        # Call the LLM to analyze the dataset
        analysis_result = analyze_dataset_with_llm(df)

        # Return the detected type along with explanation and categories (if any)
        return JsonResponse({
            'status': 'success',
            'type': analysis_result.get('type'),
            'explanation': analysis_result.get('explanation'),
            'categories': analysis_result.get('categories', [])  # Default to an empty list if not provided
        })

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)  # Log the error
        return HttpResponse(error_message, status=500)


# Getting prepared the predefined kpis with this
import os
import pandas as pd
import base64
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from plotly.graph_objs import Figure


@csrf_exempt
def predefined_kpi_getting(request):
    try:
        if request.method != "POST":
            return HttpResponse("Invalid request method. Only POST requests are allowed.", status=405)

        kpi_type = request.POST.get('type')
        kpi_category = request.POST.get('category')
        kpi_names = request.POST.getlist('kpi_name')

        if not kpi_type or not kpi_category or not kpi_names:
            return JsonResponse({
                'status': 'failure',
                'message': "Both 'type', 'category', and 'kpi_name' are required."
            })

        processed_data_path = os.path.join('uploads', 'processed_data.csv')
        if not os.path.exists(processed_data_path):
            return JsonResponse({
                'status': 'failure',
                'message': "No processed data file found."
            })

        df = pd.read_csv(processed_data_path)
        df.to_csv('data.csv', index=False)

        kpi_results = []

        for kpi_name in kpi_names:
            try:
                charts_data, kpi_code = generate_kpi_code([kpi_name])

                # Ensure data and layout are returned in the response
                kpi_charts = [
                    {
                        "data": chart["data"],
                        "layout": chart["layout"]
                    }
                    for chart in charts_data
                ]

                kpi_results.append({
                    'kpi_name': kpi_name,
                    'kpi_code': kpi_code,
                    'charts_data': kpi_charts
                })
            except Exception as e:
                kpi_results.append({
                    'kpi_name': kpi_name,
                    'error': f"Error generating code or graph for KPI '{kpi_name}': {str(e)}"
                })

        successful_kpis = [kpi for kpi in kpi_results if 'charts_data' in kpi]
        if not successful_kpis:
            return JsonResponse({
                'status': 'failure',
                'message': 'Failed to generate code or graphs for all KPIs.',
                'kpi_results': kpi_results
            })

        return JsonResponse({
            'status': 'success',
            'type': kpi_type,
            'category': kpi_category,
            'kpis': kpi_results
        })

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return JsonResponse({
            'status': 'failure',
            'message': error_message
        })


# For Model checking....
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pmdarima as pm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta
import xml.etree.ElementTree as ET
from keras.models import load_model
import matplotlib.pyplot as plt


@csrf_exempt
def models(request):
    try:
        df = pd.read_csv('data.csv')
        print("Data preview:\n", df.head(5))

        single_value_columns = [col for col in df.columns if df[col].nunique() == 1]
        df.drop(single_value_columns, axis=1, inplace=True)

        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) < 1:
            return JsonResponse({"msg": "This dataset doesn't meet the modeling requirement"}, status=400)

        if request.method == 'POST':
            print("POST data:", request.POST.dict())
            model_type = request.POST.get('model')
            col = request.POST.get('col')
            request.session['col_predict'] = col

            if not model_type or not col:
                return JsonResponse({'msg': 'Missing model or column parameter'}, status=400)

            if model_type == 'RandomForest':
                stat, cols = random_forest(df, col)
                return JsonResponse({
                    'columns': list(df.columns),
                    'rf': True,
                    'status': stat,
                    'rf_cols': cols
                })

            elif model_type == 'K-Means':
                stat, clustered_data = kmeans_train(df)
                return JsonResponse({
                    'columns': list(df.columns),
                    'cluster': True,
                    'status': stat,
                    'clustered_data': clustered_data.to_json()
                })

            elif model_type == "Arima":
                frequency = request.POST.get('frequency')
                tenure = request.POST.get('tenure')
                if not frequency or not tenure:
                    return JsonResponse({'msg': 'Missing frequency or tenure for ARIMA model'}, status=400)

                stat, data, img_data = arima_train(df, col, {
                    'time_unit': frequency,
                    'forecast_horizon': int(tenure)
                })

                return JsonResponse({
                    'columns': list(df.columns),
                    'status': stat,
                    'arima': True,
                    'path': img_data if isinstance(img_data, str) else str(img_data),
                    'data': data.to_json()
                })

            elif model_type == 'OutlierDetection':
                res = outlier_check(df, col)
                return JsonResponse({
                    'columns': list(df.columns),
                    'status': True,
                    'processed_data': markdown_to_html(res),
                    'OutlierDetection': True
                })

            return JsonResponse({'msg': 'Unsupported model type'}, status=400)

        return JsonResponse({'columns': list(df.columns)})

    except Exception as e:
        print("Error:", e)
        return JsonResponse({'msg': str(e)}, status=500)


def outlier_check(df, column):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f'detect outliers for  the following data {df[column]}'}
        ]
    )
    all_text = ""
    # Display generated content dynamically
    for choice in response.choices:
        message = choice.message
        chunk_message = message.content if message else ''
        # Replace 'red' styling keywords with 'skyblue' or similar
        chunk_message = chunk_message.replace("color: red;", "color: skyblue;")
        all_text += chunk_message
    print(all_text)
    return all_text


def find_elbow_point(inertia_values):
    # Calculate the rate of change between successive inertia values
    changes = np.diff(inertia_values)
    # Identify the elbow as the point where change starts to decrease
    elbow_point = np.argmin(np.abs(np.diff(changes))) + 1
    return elbow_point


def arima_train(data, target_col, bot_query=None):
    try:
        print('ArimaTrain')
        print("Column dtypes:\n", data.dtypes)
        # Identify date column by checking for datetime type
        date_column = None
        results = {}
        if not os.path.exists(os.path.join("models", 'Arima', target_col)):
            for col in data.columns:
                if data.dtypes[col] == 'object':
                    try:
                        # Attempt to convert column to datetime
                        pd.to_datetime(data[col])
                        date_column = col
                        break
                    except (ValueError, TypeError):
                        continue
            if not date_column:
                raise ValueError("No datetime column found in the dataset.")
            print(date_column)
            # Set the date column as index
            data[date_column] = pd.to_datetime(data[date_column])
            data.set_index(date_column, inplace=True)

            try:
                data_actual = data[[target_col]]
                data_actual.reset_index(inplace=True)
                data_actual.columns = ["datetime", 'value']
                data_actual.set_index("datetime", inplace=True)
                train_frequency = check_data_frequency(data_actual)

                train_models(data_actual, target_col)

                with open(os.path.join("models", 'Arima', target_col, target_col + '_results.json'), 'w') as fp:
                    json.dump({'data_freq': train_frequency}, fp, indent=4)
                # result_graph = plot_graph(results, os.path.join('models', 'arima', target_col))
            except Exception as e:
                print(e)

        frequency = bot_query['time_unit']
        periods = bot_query['forecast_horizon']
        model_path = os.path.join(os.getcwd(), 'models', 'Arima', target_col, frequency, "best_model.pkl")
        print("model_path", model_path)
        loaded_model = load_forecast_model(model_path)
        freq_map = {
            'hours': 'H',
            'days': 'D',
            'weeks': 'W',
            'months': 'MS',
            'quarters': 'QS',
            'years': 'YS'
        }

        forecasted_data = forecast(loaded_model, periods, freq_map[frequency])
        print(forecasted_data)

        result_graph = plot_graph(forecasted_data)

        print(f"Results saved to {os.path.join('models', 'arima', target_col, target_col + '_results.json')}")
        return True, forecasted_data, result_graph

    except Exception as e:
        print("ARIMA error:", e)
        return False, pd.DataFrame(), ""


def check_data_frequency(train):
    data_freq = {'D': 'Days', 'W': 'Weeks', "H": "Hours", "Q": "Quarters", 'A': 'Years'}
    m = pd.infer_freq(train.index)
    if m in ['15T', '30T', "H", "D", "W", "M", "Q", "A"]:
        return data_freq[m]
    else:
        print('Unsupported frequency')


# Forecast
def forecast(model, periods, freq):
    future = pd.date_range(start=pd.Timestamp.now(), periods=periods, freq=freq)
    future = future.to_series().dt.date.tolist()

    model_type = str(type(model))
    print(f"Detected model type: {model_type}")

    try:
        if 'Prophet' in model_type:
            future_df = pd.DataFrame({'ds': future})
            # Prophet expects 'ds' column and returns 'yhat'
            forecast = model.predict(future_df)
            if 'yhat' in forecast.columns:
                forecast = forecast[['ds', 'yhat']]
                forecast['yhat'] = forecast['yhat'].round(2)
                forecast.columns = ['date', 'forecasted_value']
                return forecast
            else:
                raise ValueError("Prophet output missing 'yhat' column.")

        else:
            # ARIMA, XGBoost, RandomForest — Expect direct prediction
            future_df = pd.DataFrame({'date': future})  # Rename here
            if 'ARIMA' in model_type:
                forecast = model.forecast(steps=future_df.shape[0])
                future_df['forecasted_value'] = np.round(forecast.values, 2)
            else:
                start_idx = model.last_index_ + 1  # get from model
                end_idx = start_idx + len(future_df)
                X_future = np.arange(start_idx, end_idx).reshape(-1, 1)
                print(X_future)
                forecast = model.predict(X_future)
                future_df['forecasted_value'] = np.round(forecast, 2)

            return future_df[['date', 'forecasted_value']]

    except Exception as e:
        print(f"Prediction Error: {e}")


def train_models(df, target_col):
    frequencies = ['hours', 'days', 'weeks', 'months', 'years']
    for freq in frequencies:
        print(f"\nTraining {freq} models...")

        # Resample the data for each frequency
        resampled_df = resample_data(df, freq)
        train, test = train_test_split(resampled_df, test_size=0.2, shuffle=False)

        trend = detect_trend(train)
        seasonality = detect_seasonality(train)

        best_model = None
        best_error = float('inf')
        best_model_name = ""
        scenario = ""

        # Scenario 1: Trend only
        if trend and not seasonality:
            scenario = "Trend only"
            arima_model, arima_error = train_arima(train, test)
            xgb_model, xgb_error = train_xgboost(train, test)

            if arima_error < xgb_error:
                best_model, best_error = arima_model, arima_error
                best_model_name = "ARIMA"
            else:
                best_model, best_error = xgb_model, xgb_error
                best_model_name = "XGBoost"

        # Scenario 2: Seasonality only
        if seasonality and not trend:
            scenario = "Seasonality only"
            prophet_model, prophet_error = train_prophet(train, test)
            arima_model, arima_error = train_arima(train, test)

            if prophet_error < arima_error:
                best_model, best_error = prophet_model, prophet_error
                best_model_name = "Prophet"
            else:
                best_model, best_error = arima_model, arima_error
                best_model_name = "ARIMA"

        # Scenario 3: Trend + Seasonality
        if trend and seasonality:
            scenario = "Trend + Seasonality"
            prophet_model, prophet_error = train_prophet(train, test)
            arima_model, arima_error = train_arima(train, test)

            min_error = min(prophet_error, arima_error)
            if min_error == prophet_error:
                best_model, best_error = prophet_model, prophet_error
                best_model_name = "Prophet"
            elif min_error == arima_error:
                best_model, best_error = arima_model, arima_error
                best_model_name = "ARIMA"

        # Scenario 4: No Trend or Seasonality
        if not trend and not seasonality:
            scenario = "No trend or seasonality"
            xgb_model, xgb_error = train_xgboost(train, test)
            rf_model, rf_error = train_randomforest(train, test)

            if xgb_error < rf_error:
                best_model, best_error = xgb_model, xgb_error
                best_model_name = "XGBoost"
            else:
                best_model, best_error = rf_model, rf_error
                best_model_name = "RandomForest"

        # Save the best model
        if best_model:
            model_dir = f'models/Arima/{target_col}/{freq}'
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'best_model.pkl')
            save_best_model(best_model, model_path)

            # Save Scenario with Model Name
            with open(f'scenario_{freq}.json', 'w') as f:
                json.dump({"scenario": scenario, "model_name": best_model_name}, f)

            print(f"\n{freq.capitalize()} Training complete. Scenario: {scenario}, Model: {best_model_name}")

def load_forecast_model(model_path):
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        return joblib.load(model_path)
    else:
        print(f"No model found at {model_path}")
        return None


def resample_data(df, freq):
    print(f"Resampling data to {freq} frequency")
    if freq == 'hours':
        return df.resample('H').mean().ffill()
    elif freq == 'days':
        return df.resample('D').mean().ffill()
    elif freq == 'weeks':
        return df.resample('W').mean().ffill()
    elif freq == 'months':
        return df.resample('M').mean().ffill()
    elif freq == 'years':
        return df.resample('A').mean().ffill()
    else:
        raise ValueError("Unsupported frequency")


# Check Trend using Augmented Dickey-Fuller Test
def detect_trend(df):
    print('Detecting Trend...')
    result = adfuller(df['value'])
    p_value = result[1]
    return p_value > 0.05  # If p-value > 0.05 → Trend exists


# Check Seasonality using autocorrelation
def detect_seasonality(df):
    print('Detecting Seasonality...')
    autocorr = df['value'].autocorr(lag=1)
    return abs(autocorr) > 0.3  # If autocorr > 0.3 → Seasonality exists

def train_arima(train, test):
    model = ARIMA(train['value'], order=(1, 1, 1)).fit()
    pred = model.predict(start=test.index[0], end=test.index[-1])
    error = mean_squared_error(test['value'], pred, squared=False)
    return model, error

# Train Prophet Model
def train_prophet(train, test):
    print("Training Prophet...")
    prophet_df = train.reset_index().rename(columns={'datetime': 'ds', 'value': 'y'})
    model = Prophet()
    model.fit(prophet_df)

    future = pd.DataFrame({'ds': test.index})
    forecast = model.predict(future)
    error = mean_squared_error(test['value'], forecast['yhat'], squared=False)
    return model, error


# Train XGBoost Model
def train_xgboost(train, test):
    print("Training XGBoost...")
    X_train = np.arange(len(train)).reshape(-1, 1)
    y_train = train['value'].values
    X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)

    model = XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)
    model.last_index_ = len(train) + len(test) - 1
    pred = model.predict(X_test)
    error = mean_squared_error(test['value'], pred, squared=False)
    return model, error


# Train RandomForest Model
def train_randomforest(train, test):
    print("Training RandomForest...")
    X_train = np.arange(len(train)).reshape(-1, 1)
    y_train = train['value'].values
    X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    model.last_index_ = len(train) + len(test) - 1
    pred = model.predict(X_test)
    error = mean_squared_error(test['value'], pred, squared=False)
    return model, error


# Save the Best Model
def save_best_model(model, model_path):
    joblib.dump(model, model_path)



import plotly.graph_objects as go
def plot_graph(data):
    try:

        # Create Plotly figure
        fig = go.Figure()
        try:
            data['date'] = data['date'].dt.strftime('%Y-%m-%d')
        except Exception as e:
            print(e)
        # Actual Data Line
        fig.add_trace(go.Scatter(
            x=data['date'], y=data["forecasted_value"],
            mode='lines+markers', name='Forecast',
            line=dict(color='blue'), marker=dict(symbol='circle')
        ))

        # Forecast Data Line
        # fig.add_trace(go.Scatter(
        #     x=forecast_dates, y=forecast_values,
        #     mode='lines+markers', name='Forecast',
        #     line=dict(color='orange', dash='dash'), marker=dict(symbol='x')
        # ))

        # Layout Settings
        fig.update_layout(
            title=f'Forecast Values Over Time',
            xaxis_title='Date',
            yaxis_title='Values',
            xaxis=dict(tickangle=-45, type='category', tickformat='%Y-%m-%d'),
            template="plotly_white",
            width=1000, height=600
        )

        # Convert figure to Base64 Image
        fig.show()
        return make_serializable(fig.to_json())

    except Exception as e:
        print(e)
        return str(e)


def kmeans_train(data):
    try:
        # Identify categorical and numerical columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        # Handle missing values (if any)
        imputer = SimpleImputer(strategy='mean')
        data[numerical_columns] = imputer.fit_transform(data[numerical_columns])
        joblib.dump(imputer, 'imputer.pkl')

        # Build a transformer for preprocessing: scaling numerical columns and encoding categorical columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_columns),  # Standard scaling for numerical columns
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
                # One-Hot encoding for categorical columns
            ])

        # Apply preprocessing and fit KMeans
        X = preprocessor.fit_transform(data)

        # Find the optimal k using the elbow method with KMeans
        inertia = []
        K_range = range(1, 11)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)

        # Determine the optimal k
        optimal_k = find_elbow_point(inertia)
        print('Optimal number of clusters (k) based on the Elbow Method:', optimal_k)

        # Initialize KMeans with the optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_k, random_state=0)

        # Fit KMeans to the preprocessed data
        kmeans.fit(X)

        # Save the trained model and preprocessor
        joblib.dump(kmeans, 'kmeans_model.pkl')  # Save KMeans model
        joblib.dump(preprocessor, 'preprocessor.pkl')  # Save Preprocessing pipeline

        # Add cluster labels to the original data
        data['Cluster'] = kmeans.labels_
        return True, data
    except Exception as e:
        print(e)
        return False, data


def load_pipeline(save_path="model_pipeline.pkl"):
    # Load the saved pipeline
    pipeline = joblib.load(save_path)
    print(f"Pipeline loaded from: {save_path}")
    return pipeline


def random_forest(data, target_column):
    try:
        if not os.path.exists(os.path.join("models", "rf", target_column, 'deployment.json')):
            os.makedirs(os.path.join("models", "rf", target_column), exist_ok=True)
            # Separate features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Detect categorical and numerical features
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

            # Preprocessing pipelines for numerical and categorical data
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])

            # Combine preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ])

            # Choose Random Forest type based on target type
            if y.nunique() <= 5:  # Classification for few unique target values
                model_type = 'Classification'
                model = RandomForestClassifier(random_state=42)
            else:  # Regression for continuous target values
                model_type = 'Regression'
                model = RandomForestRegressor(random_state=42)

            # Create pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the pipeline
            pipeline.fit(X_train, y_train)

            cv = min(5, len(X_test))

            # Evaluate the model using cross-validation
            scores = cross_val_score(pipeline, X_test, y_test, cv=cv)
            print(f"Model Performance (CV): {scores.mean():.4f} ± {scores.std():.4f}")

            # Save the pipeline
            joblib.dump(pipeline, os.path.join("models", "rf", target_column, "pipeline.pkl"))
            print(f'Pipeline saved to: {os.path.join("models", "rf", target_column, "pipeline.pkl")}')

            with open(os.path.join("models", "rf", target_column, "deployment.json"), "w") as fp:
                json.dump({"columns": list(X_train.columns), "model_type": model_type, "Target_column": target_column},
                          fp, indent=4)
            return True, list(X_train.columns)
        else:
            with open(os.path.join(os.getcwd(), "models", "rf", target_column, 'deployment.json'), "r") as fp:
                data = json.load(fp)
            return True, data['columns']
    except Exception as e:
        print(e)
        return False, []


# Model prediction for random forest
from django.http import JsonResponse, HttpResponse
import os
import pandas as pd
from joblib import load
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def model_predict(request):
    try:
        if request.POST.get('form_name') == 'rf':
            res = {}
            for col in request.POST:
                if col == "targetColumn":
                    targetcol = request.POST[col]
                    continue
                res.update({col: request.POST[col]})
            del res['form_name']
            df = pd.DataFrame([res])
            loaded_pipeline = load_pipeline(
                os.path.join("models", "rf", targetcol, "pipeline.pkl"))
            predictions = loaded_pipeline.predict(df)
            print(predictions)
            return JsonResponse(
                {
                    'columns': list(df.columns),
                    'rf_result': f"Predicted {targetcol} value is {round(predictions[0], 2)}"
                }
            )

    except Exception as e:
        print(e)
        return JsonResponse(
            {
                'columns': [],
                'rf_result': "NA"
            }
        )


##Visualisation updated for both text and graph responses:
from rest_framework.response import Response
from rest_framework import status


@csrf_exempt
def gen_ai_bot(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed.'}, status=405)

    try:
        df = pd.read_csv('data.csv')

        metadata_str = ", ".join(df.columns.tolist())
        sample_data = df.head(2).to_dict(orient='records')

        # Handle both form-encoded and JSON payloads
        try:
            body = json.loads(request.body)
            prompt = body.get('prompt')
        except json.JSONDecodeError:
            prompt = request.POST.get('prompt')

        if not prompt:
            return JsonResponse({'error': 'Prompt is required.'}, status=400)

        if 'forecast' in prompt.lower():
            data = extract_forecast_details_llm(prompt, df.columns)
            stat, data, img_data = arima_train(df, data['target_variable'], data)

            return JsonResponse({
                'data': data.to_json(),
                'plot': make_serializable(img_data)
            }, status=200)
        else:
            system_prompt = f"""You are an AI specialized in data analytics and visualization. The data for analysis is 
            stored in a CSV file named data.csv, with the following attributes: {metadata_str} and sample data as 
            {sample_data}.

            Follow these rules while responding to user queries:

            1. Strictly use 'data.csv' as the data source without stating any limitations or disclaimers about file access.
            2. Data Analysis: If the query requires numerical or tabular insights, extract relevant data from 
            data.csv, perform necessary calculations, and provide a concise summary. Store the result in text_output.
            3. Visualization: If the query requires a graph, generate Python code using Plotly with fig as output.
            4. Forecasting: Generate forecast using ARIMA and store results in text_output and plot in fig.
            """

            result = {}
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )

            pre_code_text, post_code_text, code = process_genai_response(response)
            result.update({
                'text_pre_code_response': pre_code_text,
                'code': code,
                'text_post_code_response': post_code_text
            })

            if 'import' in code:
                namespace = {}
                try:
                    exec(code, namespace)
                    result['text_output'] = namespace.get('text_output')

                    fig = namespace.get('fig')
                    if fig and isinstance(fig, Figure):
                        result['chart_response'] = make_serializable(fig.to_plotly_json())

                except Exception as e:
                    return JsonResponse({'message': str(e)}, status=500)

            return JsonResponse(result, status=200)

    except Exception as e:
        return JsonResponse({'message': str(e)}, status=500)



def extract_forecast_details_llm(prompt, column_names):
    try:
        system_prompt = f""" You are an AI assistant that extracts forecast details from a user's prompt. Given a 
        natural language input and the following column names from the input data, return the following in JSON format:

            1. "target_variable" - The thing being forecasted (e.g., "sales", "revenue"). - If the target variable is 
            misspelled or ambiguous, try to match it to the closest column name from the list below. 2. 
            "forecast_horizon" - The number of time steps. 3. "time_unit" - The unit of time (days, months, years).

            Available column names: {', '.join(column_names)}

            Example Outputs:
            - Input: "Forecast the sales data for 5 years."
              Output: {{"target_variable": "sales", "forecast_horizon": 5, "time_unit": "years"}}

            - Input: "Can you predict electricity demand for the next 12 months?"
              Output: {{"target_variable": "electricity demand", "forecast_horizon": 12, "time_unit": "months"}}

            - Input: "I want to predict CO2 levels for 7 days."
              Output: {{"target_variable": "CO2 levels", "forecast_horizon": 7, "time_unit": "days"}}

            Ensure that the "target_variable" matches one of the available column names, even if the user misspells it.
            """
        forecast_details = ''
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0  # Make it deterministic
        )
        for choice in response.choices:
            message = choice.message
            chunk_message = message.content if message else ''
            forecast_details += chunk_message
        print(forecast_details)

        return eval(forecast_details)

    except Exception as e:
        print(e)


def process_genai_response(response):
    all_text = ""
    text_post_code = ''
    code_start = -1
    code_end = -1
    for choice in response.choices:
        message = choice.message
        chunk_message = message.content if message else ''
        all_text += chunk_message
    print(all_text)
    if "```python" in all_text:
        code_start = all_text.find("```python") + 9
        code_end = all_text.find("```", code_start)
        code = all_text[code_start:code_end]
    else:
        code = all_text
    text_pre_code = all_text[:code_start - 9]
    if code_start != -1:
        text_post_code = all_text[code_end:]
    return text_pre_code, text_post_code, code


def make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


# Dashboard with AI
from plotly.graph_objects import Figure
import pandas as pd

def analyze_dataset1(df):
    global important_numerical, important_categorical
    queries = []

    # Get metadata about the dataset
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()
    date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]

    # Select top 3-4 important numerical columns (based on correlation or variance)
    if numerical_columns:
        # Calculate correlation to find relationships
        correlation_matrix = df[numerical_columns].corr().abs()
        important_numerical = correlation_matrix.mean().nlargest(3).index.tolist()  # Top 3 numerical columns

    # Select the most important categorical column (based on unique values)
    if categorical_columns:
        important_categorical = max(categorical_columns, key=lambda col: df[col].nunique())

    # BASIC GRAPHS (4 total)
    # 1. Histogram (basic)
    if important_numerical:
        queries.append({
            "type": "histogram",
            "query": f"Generate a histogram for '{important_numerical[0]}' to show value distribution.",
            "analysis": "Basic Distribution Analysis",
            "category": "basic"
        })

    # 2. Bar chart (basic)
    if categorical_columns:
        queries.append({
            "type": "bar",
            "query": f"Generate a bar chart showing value counts for '{important_categorical}'.",
            "analysis": "Basic Categorical Analysis",
            "category": "basic"
        })

    # 3. Scatter plot (basic)
    if len(important_numerical) >= 2:
        queries.append({
            "type": "scatter",
            "query": f"Generate a scatter plot comparing '{important_numerical[0]}' and '{important_numerical[1]}'.",
            "analysis": "Basic Relationship Analysis",
            "category": "basic"
        })

    # 4. Line chart (basic)
    if date_columns and important_numerical:
        queries.append({
            "type": "line",
            "query": f"Generate a line chart showing '{important_numerical[0]}' over time using '{date_columns[0]}'.",
            "analysis": "Basic Trend Analysis",
            "category": "basic"
        })
    elif important_numerical:  # Fallback if no date columns
        queries.append({
            "type": "line",
            "query": f"Generate a line chart showing trend of '{important_numerical[0]}'.",
            "analysis": "Basic Trend Analysis",
            "category": "basic"
        })

    # ADVANCED GRAPHS (4 total)
    # 1. Box plot (advanced)
    if important_numerical:
        queries.append({
            "type": "box",
            "query": f"Generate a box plot for '{important_numerical[0]}' to analyze outliers and data distribution.",
            "analysis": "Advanced Outlier Detection",
            "category": "advanced"
        })

    # 2. Heatmap (advanced)
    if len(numerical_columns) >= 2:
        queries.append({
            "type": "heatmap",
            "query": f"Generate a heatmap showing correlations between numerical columns.",
            "analysis": "Advanced Correlation Analysis",
            "category": "advanced"
        })

    # 3. Violin plot (advanced)
    if important_numerical and categorical_columns:
        queries.append({
            "type": "violin",
            "query": f"Generate a violin plot comparing distribution of '{important_numerical[0]}' across categories in '{important_categorical}'.",
            "analysis": "Advanced Distribution Comparison",
            "category": "advanced"
        })

    # 4. 3D Scatter plot (advanced)
    if len(important_numerical) >= 3:
        queries.append({
            "type": "3d_scatter",
            "query": f"Generate a 3D scatter plot analyzing relationships between '{important_numerical[0]}', '{important_numerical[1]}', and '{important_numerical[2]}'.",
            "analysis": "Advanced Multivariate Analysis",
            "category": "advanced"
        })

    elif len(important_numerical) >= 2:  # Fallback if only 2 numerical columns
        queries.append({
            "type": "scatter_matrix",
            "query": f"Generate a scatter matrix for numerical columns to analyze pairwise relationships.",
            "analysis": "Advanced Pairwise Analysis",
            "category": "advanced"
        })

    if important_numerical:
        queries.append({
            "type": "distplot",
            "query": f"Generate a distplot showing the distribution of '{important_numerical[0]}'.",
            "analysis": f"Distribution Analysis",
            "category": "advanced"
        })

    return queries

@csrf_exempt
def gen_plotly_response(request):
    if request.method == "POST":
        try:
            # Load CSV
            csv_file_path = 'data.csv'
            df = pd.read_csv(csv_file_path)

            # Convert date columns to datetime if applicable
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            for col in date_columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

            # Analyze the dataset and generate meaningful queries
            queries = analyze_dataset1(df)
            print("Queries are.............................................", queries)
            if not queries:
                return JsonResponse({"message": "No meaningful queries could be generated for the dataset."},
                                    status=400)

            # Generate CSV metadata
            csv_metadata = {"columns": df.columns.tolist()}
            metadata_str = ", ".join(csv_metadata["columns"])

            # List to store all generated graphs
            all_charts = []

            for query in queries:
                # Prompt engineering for AI
                print(query)
                prompt_eng = (
                    f"You are an AI specialized in data analytics and visualization."
                    f"Data used for analysis is stored in a CSV file named 'data.csv'."
                    f"Attributes of the data are: {metadata_str}."
                    f"Consider 'data.csv' as the data source for any analysis."
                    f"Based on the user's query, generate Python code using Plotly to create the requested type of graph."
                    f"Every graph must include a title, axis labels (if applicable), and appropriate colors for better visualization."
                    f"Ensure the graph is visually appealing and provides sufficient context for understanding."
                    f"The graph must have a white background for both the plot and paper."
                    f"The code must output a Plotly 'Figure' object stored in a variable named 'fig'."
                    f"The user asks: {query}"

                )

                # Call AI to generate the code
                chat = generate_code(prompt_eng)
                print(f"Generated code for query '{query}':")
                print(chat)

                # Check for valid Plotly code in the AI response
                if 'import' in chat:
                    namespace = {}
                    try:
                        # Execute the generated code
                        exec(chat, namespace)

                        # Retrieve the Plotly figure from the namespace
                        fig = namespace.get("fig")

                        if fig and isinstance(fig, Figure):
                            # Convert the Plotly figure to JSON
                            chart_data = fig.to_plotly_json()

                            # Ensure JSON serialization by converting NumPy arrays to lists
                            def make_serializable(obj):
                                if isinstance(obj, np.ndarray):
                                    return obj.tolist()
                                elif isinstance(obj, dict):
                                    return {k: make_serializable(v) for k, v in obj.items()}
                                elif isinstance(obj, list):
                                    return [make_serializable(v) for v in obj]
                                return obj

                            # Recursively process the chart_data
                            chart_data_serializable = make_serializable(chart_data)

                            # Append the graph data to the list
                            all_charts.append(chart_data_serializable)
                        else:
                            print(f"No valid Plotly figure found for query: {query}")
                    except Exception as e:
                        error_message = f"There was an error while executing the code for query '{query}': {str(e)}"
                        print(error_message)
                else:
                    print(f"Invalid AI response for query: {query}")

            # Return all generated graphs to the frontend
            return JsonResponse({"charts": all_charts}, status=200)
        except Exception as e:
            # Handle general exceptions
            error_message = f"An unexpected error occurred: {str(e)}"
            print(error_message)
            return JsonResponse({"message": error_message}, status=500)

    # Return a fallback HttpResponse for invalid request methods
    return HttpResponse("Invalid request method", status=405)


# Column_description for the  Discover in UI
@csrf_exempt
def col_description(request):
    if request.method == "POST":
        csv_file_path = 'data.csv'
        df = pd.read_csv(csv_file_path)
        print(df.head(5))

        prompt_eng = (
            f"You are analytics_bot. Analyse the data: {df.head()} and give description of the columns"
            f"Just provide the column name and the description regarding the column name in the next line."

        )
        column_description = generate_code(prompt_eng)

        return JsonResponse({"Column_description": markdown_to_html(column_description)})


# Hana bot File uploading:
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .Hana_bot import HanaBOT

# Define the directory where uploaded files will be stored
UPLOAD_DIR = "chat_to_doc"

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


@csrf_exempt
def upload_and_process_file(request):
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.FILES:
            return JsonResponse({"error": "No file uploaded"}, status=400)

        uploaded_file = request.FILES['file']

        # Save the uploaded file locally
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # Initialize HanaBOT
        bot = HanaBOT(index_path="faiss_index")

        # Process the file
        file_extension = uploaded_file.name.split(".")[-1].lower()
        try:
            texts = bot.load_file(file_path, file_extension)
            storing = bot.process_and_store(texts)
            print("Text and storing,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,", storing)
            # Optionally, you can remove the file after processing if it's no longer needed
            # os.remove(file_path)
            return JsonResponse({"message": "File processed successfully!"}, status=200)
        except Exception as e:
            # Optionally, you can remove the file if an error occurs
            # os.remove(file_path)
            return JsonResponse({"error": str(e)}, status=400)
    else:
        return JsonResponse({"error": "Only POST requests are allowed"}, status=405)


# Hana File querying based on the above data
@csrf_exempt
def query_data(request):
    if request.method == 'POST':
        query = request.POST.get('query', '')
        greetings = {"hi", "hello", "hey", "greetings"}

        if query in greetings:
            greeting_prompt = "Respond to the user greeting in a friendly and engaging manner."
            greeting_response = generate_coding_hi(greeting_prompt)
            return JsonResponse({"answer": markdown_to_html(greeting_response)})

        else:
            # Initialize HanaBOT
            bot = HanaBOT(index_path="faiss_index")
            try:
                relevant_docs = bot.retrieve_relevant_docs(query, k=5)
                print(relevant_docs)
                answer = bot.generate_answer(query, relevant_docs)
                return JsonResponse({
                    # "relevant_docs": relevant_docs,
                    "answer": markdown_to_html(answer)
                }, status=200)
            except Exception as e:
                return JsonResponse({"error": str(e)}, status=400)
    else:
        return JsonResponse({"error": "Only POST requests are allowed"}, status=405)


def generate_coding_hi(prompt_eng):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant providing actionable insights."},
            {"role": "user", "content": prompt_eng}
        ]
    )
    return response.choices[0].message.content.strip()


# Filling missed data api
@csrf_exempt
def missing_data(request):
    if request.method == 'POST':
        csv_file_path = 'data.csv'
        df = pd.read_csv(csv_file_path)
        print(df.head(5))

        new_df, html_df = process_missing_data(df.copy())
        new_df.to_csv(os.path.join('uploads', 'processed_data.csv'), index=False)
        new_df.to_csv("data.csv", index=False)
        with open(os.path.join('mvt_data.json'), 'w') as fp:
            json.dump({'data': html_df}, fp, indent=4)

        return JsonResponse({"df": html_df})


def process_missing_data(df):
    df = convert_to_datetime(df)
    df, html_df = handle_missing_data(df)
    return df, html_df


def convert_to_datetime(df):
    """
    Converts object (string) columns containing dates to datetime format.
    """
    for col in df.columns:
        if df[col].dtype == "object":  # Process only string columns
            if df[col].str.contains(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", na=False).any():
                df[col] = df[col].apply(detect_and_parse_date)

    return df


import dateutil.parser


def detect_and_parse_date(value):
    """
    Detects and converts dates in multiple formats, including:
    - MM-DD-YYYY
    - DD-MM-YYYY
    - MM/DD/YYYY
    - DD/MM/YYYY
    - YYYY-MM-DD
    """
    if pd.isna(value) or not isinstance(value, str) or value.strip() == "":
        return pd.NaT  # Handle missing values safely

    try:
        # Check if it's a date with hyphens or slashes
        if re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}$", value):
            day_first = False  # Assume MM-DD-YYYY first

            # Check for an ambiguous case (day > 12) → Must be DD-MM-YYYY
            parts = re.split(r"[-/]", value)
            month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
            if day > 12:
                day_first = True  # Switch to DD-MM-YYYY

            # Parse with detected format
            return dateutil.parser.parse(value, dayfirst=day_first)

        # Otherwise, use default dateutil parsing
        return dateutil.parser.parse(value)

    except ValueError:
        return pd.NaT  # Return NaT if parsing fails


def handle_missing_data(df):
    try:
        # Identify numeric and datetime columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        date_time_cols = df.select_dtypes(include=['datetime64']).columns

        # Impute numeric columns and track which cells were imputed
        imputer = KNNImputer(n_neighbors=5)
        imputed_numeric = imputer.fit_transform(df[numeric_cols])
        imputed_numeric_df = pd.DataFrame(imputed_numeric, columns=numeric_cols).round(2)

        # Mark imputed cells (True if the original cell was NaN)
        imputed_flags = df[numeric_cols].isnull()
        imputed_flags = imputed_flags.applymap(lambda x: x if x else False)

        # Update DataFrame with imputed values
        df[numeric_cols] = imputed_numeric_df

        # Handle datetime columns by forward filling missing values
        for col in date_time_cols:
            df[col] = pd.to_datetime(df[col])
            time_diffs = df[col].diff().dropna()
            avg_diff_sec = time_diffs.mean().total_seconds()
            minute_sec = 60
            hour_sec = 3600
            day_sec = 86400
            month_sec = day_sec * 30.44
            year_sec = day_sec * 365.25

            if avg_diff_sec < hour_sec:
                time_unit = "minutes"
                avg_diff = pd.Timedelta(minutes=avg_diff_sec / minute_sec)
            elif avg_diff_sec < day_sec:
                time_unit = "hours"
                avg_diff = pd.Timedelta(hours=avg_diff_sec / hour_sec)
            elif avg_diff_sec < month_sec:
                time_unit = "days"
                avg_diff = pd.Timedelta(days=avg_diff_sec / day_sec)
            elif avg_diff_sec < year_sec:
                time_unit = "months"
                avg_diff = pd.DateOffset(months=round(avg_diff_sec / month_sec))
            else:
                time_unit = "years"
                avg_diff = pd.DateOffset(years=round(avg_diff_sec / year_sec))

            for i in range(1, len(df)):
                if pd.isnull(df[col].iloc[i]):
                    df.loc[i, col] = df[col].iloc[i - 1] + avg_diff
                    imputed_flags.loc[i, col] = True

            imputed_flags.fillna(False, inplace=True)

        # Convert the DataFrame into a JSON-serializable format with flags
        data = []
        for _, row in df.iterrows():
            row_data = {}
            for col in df.columns:
                row_data[col] = {
                    "value": row[col].strftime('%Y-%m-%d %H:%M:%S') if isinstance(row[col], pd.Timestamp) else row[col],
                    "is_imputed": str(imputed_flags[col].get(_, False)) if col in imputed_flags else str(False)
                    # Check if cell was imputed
                }
            data.append(row_data)
        return df, data
    except Exception as e:
        print(e)


###Data scout Apis:
from .data_scout import DataScout_agent,DataScout_agent_with_pdf
@csrf_exempt
@api_view(['POST'])
def create_data_with_data_scout(request):
    prompt = request.data.get('prompt')
    data_type = request.data.get('type')

    if not prompt or not data_type:
        return Response({"error": "Prompt and type are required"}, status=400)

    if data_type == "Excel":
        agent1 = DataScout_agent()
    else:
        agent1 = DataScout_agent_with_pdf()

    try:
        result = agent1.invoke(prompt)
        if 'output' in result:
            return Response({"file_path": result['output']})
        else:
            return Response({"error": "Failed to generate file"}, status=500)
    except Exception as e:
        return Response({"error": str(e)}, status=500)


#Predictive Maintenence Apis:
# @csrf_exempt
# def predictive_maintenence(request):
#     if request.method == "POST":
#         try:
#             # Load CSV
#             csv_file_path = 'data.csv'
#             df = pd.read_csv(csv_file_path)
