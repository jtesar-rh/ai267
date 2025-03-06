from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset, Model, importer

DSI = 'quay.io/modh/cuda-notebooks:cuda-jupyter-tensorflow-ubi9-python-3.11-20250213-b23e7ed'

@dsl.component( base_image=DSI)
def gather_data(data: Output[Dataset]):
    import pandas as pd
    import os
    import io
    import boto3
    import numpy as np

    s3 = boto3.client(
        "s3",
        'us-east-1',
        aws_access_key_id='minio',
        aws_secret_access_key='minio123',
        endpoint_url='https://minio-api-minio.apps.ocp4.example.com'
    )

    objects = s3.list_objects_v2(Bucket='bostonhousing')

    for obj in objects["Contents"]:
        print(obj["Key"])
        s3.download_file('bostonhousing', obj["Key"], obj["Key"], )
       
        ds = pd.read_csv('boston_housing.csv')
        ds.to_csv(data.path)



@dsl.component( base_image=DSI)
def clean_data(data_in: Input[Dataset], data_out: Output[Dataset]):
    import pandas as pd
    df = pd.read_csv(data_in.path)
    nhdi = df[(df['PRICE'] > 45) | (df['RM'] < 4)].index
    df.drop(nhdi)
    print('---------------------- DATA -------------------------')
    print(data_out.uri)
    print(data_out.path)
    print('---------------------- END DATA -------------------------')
    df.to_csv(data_out.path)


@dsl.component( base_image=DSI)
def train(data: Input[Dataset], model_out: Output[Model]):
    import pandas as pd
    import tensorflow as tf
    import numpy as np

    df = pd.read_csv(data.path)
    x_train,y_train = df.RM.values[:-100], df.PRICE.values[:-100]
    x_test,y_test = df.RM.values[-100:], df.PRICE.values[-100:]
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(1,)))
    model.add(tf.keras.layers.Normalization())
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='rmsprop',loss=tf.keras.losses.MeanSquaredError(),metrics=[tf.keras.metrics.RootMeanSquaredError()])
    history = model.fit(x_train,y_train,validation_split=0.1,epochs=300,verbose=2)
    
    model_out.uri = model_out.uri + '.keras'
    model.save(model_out.path)

@dsl.component( base_image=DSI)
def infer(model_in: Input[Model]):
    import numpy as np
    import tensorflow as tf

    model = tf.keras.models.load_model(model_in.path)

    x = tf.linspace(4,12,10)
    y = model.predict(x)

    print('-------------------- Predictions ------------------------')
    print(y)


@dsl.pipeline(name='boston-housing')
def boston_pipeline():
  
   importer1 = importer(artifact_uri='s3://pipbucket/data/boston_housing.csv',artifact_class=Dataset,reimport=False)    
   #task1 = gather_data()
   task2 = clean_data(data_in = importer1.output)
   task3 = train(data = task2.outputs['data_out'])
   infer(model_in = task3.outputs['model_out'])


compiler.Compiler().compile(boston_pipeline, "pipeline.yaml")    
