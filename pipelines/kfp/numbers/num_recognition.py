from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset, Model, importer

DSI = 'registry.access.redhat.com/ubi9/python-39'

@dsl.component(base_image=DSI,packages_to_install=["tensorflow==2.15.1","dill"])
def get_training_data(train_images: Output[Dataset],
                      train_labels: Output[Dataset],
                      test_images: Output[Dataset], 
                      test_labels: Output[Dataset]):
    import numpy as np
    import tensorflow as tf
    
    mnist = tf.keras.datasets.mnist
    (tr_i, tr_l), (t_i, t_l) = mnist.load_data()

    train_images.uri += '.npy'
    train_labels.uri += '.npy'
    test_images.uri +=  '.npy'
    test_labels.uri +=  '.npy'

    tr_i = tr_i / 255.0
    t_i = t_i / 255.0

    np.save(train_images.path,tr_i)
    np.save(train_labels.path,tr_l)
    np.save(test_images.path,t_i)
    np.save(test_labels.path,t_l)


@dsl.component(base_image=DSI,packages_to_install=["tensorflow==2.15.1","dill"])
def define_model(model_out: Output[Model]):
    import tensorflow as tf
    
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28),name='bitmap'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')])    

    model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    model_out.uri = model_out.uri + ".keras"
    model.save(model_out.path)

@dsl.component(base_image=DSI,packages_to_install=["tensorflow==2.15.1","dill"])
def train_model(train_images_in: Input[Dataset],
                train_labels_in: Input[Dataset],
                model_in: Input[Model],
                model_out: Output[Model]):
    import numpy as np
    import tensorflow as tf


    train_images = np.load(train_images_in.path)
    train_labels = np.load(train_labels_in.path)
    model = tf.keras.models.load_model(model_in.path)


    history = model.fit(train_images, train_labels, epochs=10,batch_size=1000,verbose=1) 
    model_out.uri = model_out.uri + '.keras'
    model.save(model_out.path)

@dsl.component(base_image=DSI,packages_to_install=["tensorflow==2.15.1","dill"])
def evaluate_model(test_images_in: Input[Dataset], 
                   test_labels_in: Input[Dataset],
                   model_in: Input[Model]) -> float:
    import numpy as np
    import tensorflow as tf

    test_images = np.load(test_images_in.path)
    test_labels = np.load(test_labels_in.path)
    model = tf.keras.models.load_model(model_in.path)

    loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('Accuracy:',str(acc))
    return acc


@dsl.component(base_image="registry.access.redhat.com/ubi9/python-39",packages_to_install=["tensorflow==2.15.1","boto3","tf2onnx==1.16.1","dill"])
def deploy_model(model_in: Input[Model],version: str) -> bool:
    import numpy as np
    import tensorflow as tf
    import boto3
    import tf2onnx
    import onnx

    model = tf.keras.models.load_model(model_in.path)

    input_signature = [tf.TensorSpec(model.inputs[0].shape, tf.float64, name='bitmap')]

    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
    onnx.save(onnx_model, "model.onnx")

    s3 = boto3.client("s3",
                      "us-east-1",
                      aws_access_key_id="minio",
                      aws_secret_access_key="minio123",
                      endpoint_url="https://minio-api-minio.apps.ocp4.example.com",
                      use_ssl=True)
    s3_path = "/deploy/numbers/" + version + "/model.onnx"
    s3.upload_file("model.onnx","numbers",s3_path)
    return True

@dsl.component(base_image="quay.io/openshift-release-dev/ocp-v4.0-art-dev@sha256:f692e2703b699f7d23e4085599d80b8db1a57196d9a3b6a5a12bdeec493d2a63")
def restart_model_server():
    import os
    os.system('oc whoami')
    os.system('oc rollout restart deployment/modelmesh-serving-numbers')

@dsl.pipeline(name='Numbers')
def numbers(model_version: str):

   train_data = get_training_data()
   model = define_model()

   model = train_model(train_images_in=train_data.outputs['train_images'],
                       train_labels_in=train_data.outputs['train_labels'],
                       model_in=model.outputs['model_out'])

   evaluation = evaluate_model(test_images_in=train_data.outputs['test_images'],
                               test_labels_in=train_data.outputs['test_labels'],
                               model_in = model.outputs['model_out'])
   with dsl.If(evaluation.output > 0.95):    
     model_deployed = deploy_model(model_in = model.outputs['model_out'],version = model_version)
     with dsl.If(model_deployed.output == True):
        restart_model_server().set_caching_options(False)



compiler.Compiler().compile(numbers, "pipeline.yaml")    
