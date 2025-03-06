from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset, Model

DSI = 'quay.io/modh/cuda-notebooks:cuda-jupyter-tensorflow-ubi9-python-3.11-20250213-b23e7ed'

@dsl.component(base_image=DSI)
def get_training_data(train_images: Output[Dataset],
                      train_labels: Output[Dataset],
                      test_images: Output[Dataset], 
                      test_labels: Output[Dataset]):
    import numpy as np
    import tensorflow as tf
    import joblib
    
    mnist = tf.keras.datasets.mnist
    (tr_i, tr_l), (t_i, t_l) = mnist.load_data()

    tr_i = tr_i / 255.0
    t_i = t_i / 255.0

    joblib.dump(tr_i,filename=train_images.path)
    joblib.dump(tr_l,train_labels.path)
    joblib.dump(t_i,test_images.path)
    joblib.dump(t_l,test_labels.path)


@dsl.component(base_image=DSI)
def define_model(model_out: Output[Model]):
    import joblib
    import tensorflow as tf

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(50,activation='relu'))
    model.add(tf.keras.layers.Dense(50,activation='relu'))
    model.add(tf.keras.layers.Dense(10,activation='softmax'))
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    joblib.dump(model,model_out.path)

@dsl.component(base_image=DSI)
def train_model(train_images_in: Input[Dataset],
                train_labels_in: Input[Dataset],
                model_in: Input[Model],
                model_out: Output[Model]):
    import numpy as np
    import tensorflow as tf
    import joblib


    train_images = joblib.load(train_images_in.path)
    train_labels = joblib.load(train_labels_in.path)
    model = joblib.load(model_in.path)


    history = model.fit(train_images, train_labels, epochs=2,batch_size=12000,verbose=1) 
    joblib.dump(model,model_out.path)

@dsl.component(base_image=DSI)
def evaluate_model(test_images_in: Input[Dataset], 
                   test_labels_in: Input[Dataset],
                   model_in: Input[Model]) -> float:
    import numpy as np
    import tensorflow as tf
    import joblib

    test_images = joblib.load(test_images_in.path)
    test_labels = joblib.load(test_labels_in.path)
    model = joblib.load(model_in.path)

    loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('Accuracy:',str(acc))
    return acc


@dsl.component(base_image=DSI)
def init_acc() -> float:
    return 0.0

@dsl.pipeline(name='Numbers')
def numbers():
   train_data = get_training_data()
   model = define_model()

   acc = init_acc().output
   with dsl.ParallelFor(
        items=[1,2,3,4,5,6,7,8,9,10],
        parallelism=1
   ) as epochs:
       with dsl.If(acc < 0.9):
         model = train_model(train_images_in=train_data.outputs['train_images'],
                             train_labels_in=train_data.outputs['train_labels'],
                             model_in=model.outputs['model_out'])

         evaluation = evaluate_model(test_images_in=train_data.outputs['test_images'],
                                     test_labels_in=train_data.outputs['test_labels'],
                                     model_in = model.outputs['model_out'])
         acc = evaluation.output
         model = model.outputs['model_out']
        



compiler.Compiler().compile(numbers, "pipeline.yaml")    
