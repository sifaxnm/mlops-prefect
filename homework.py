import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

from datetime import datetime

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df


@task
def prepare_features(df, categorical, train=True):
    # Using Prefect logger instead of print (logger.info)
    logger = get_run_logger()
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task
def train_model(df, categorical):
    # Using Prefect logger instead of print (logger.info)
    logger = get_run_logger()
    
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    # Using Prefect logger instead of print (logger.info)
    logger = get_run_logger()
    
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date):
    
    # If date is None get the current date
    if date is None:
        date = datetime.now()
    # Else get the date passed in parameter
    else:
        date = datetime.strptime(date, '%Y-%m-%d')
        
    # Get the month number
    month = date.month  
    
    # Dictionnary to assign to every month the 2 previous months in order
    # to get the appropriate parquet (or csv) files for both training and
    # validation
    dict = {
        "1": ["11","12"],
        "2": ["12","01"],
        "3": ["01","02"],
        "4": ["02","03"],
        "5": ["03","04"],
        "6": ["04","05"],
        "7": ["05","06"],
        "8": ["06","07"],
        "9": ["07","08"],
        "10": ["08","09"],
        "11": ["09","10"],
        "12": ["10","11"],
    }
    
    month_attributes = dict[str(month)]
    
    train_path = "./data/fhv_tripdata_2021-"+ month_attributes[0] +".parquet"
    val_path = "./data/fhv_tripdata_2021-"+ month_attributes[1] +".parquet"
    
    
    return train_path, val_path


@flow
def main(date=None):
    
    train_path, val_path = get_paths(date).result()
    
    print(train_path)

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)
    
    model_name = "model-"+ date +".pkl"
    dv_name = "dv-"+ date +".pkl"
    
    # Save the model
    with open("models/"+ model_name, "wb") as f_out:
        pickle.dump(lr, f_out)
        
    
    # Save the DictVectorizer
    with open("models/"+ dv_name, "wb") as f_out:
        pickle.dump(dv, f_out)

main(date="2021-08-15")

# Create a deployment with CronSchedule

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

DeploymentSpec(
    flow=main,
    name="model_training",
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="America/New_York"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)
