from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset

@dsl.component(
        base_image="image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/tensorflow"
        )
def gather_data(data: Output[Dataset]):
    import pandas as pd
    ds = pd.read_csv('boston_housing.csv')
    ds.to_csv(data.path)

@dsl.component(
        base_image="image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/tensorflow"
        )
def clean_data(data_in: Input[Dataset], data_out: Output[Dataset]):
    import pandas as pd
    ds = pd.read_csv(data_in.path)
    ds.to_csv(data_out.path)

@dsl.component(
        base_image="image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/tensorflow"
        )
def train(data: Input[Dataset]):
    import pandas as pd
    ds = pd.read_csv(data.path)

@dsl.component(
        base_image="image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/tensorflow"
        )
def infer():
    print ('Infering')

@dsl.pipeline(name='boston')
def boston_pipeline():
    gather_data(data)
    clean_data(data, clean_data)
    train(data)



compiler.Compiler().compile(boston_pipeline, "pipeline.yaml")    
