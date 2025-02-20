from kfp import dsl, compiler
from kfp.dsl import Input, Output

@dsl.component(
        base_image="image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/tensorflow"
        )
def gather_data() -> str:
    print ('Gathering data')
    data = "Hello world"
    return data

@dsl.component(
        base_image="image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/tensorflow"
        )
def clean_data(inp: str = 'dd'):
    print(data)
    print ('Cleaning data')

@dsl.component(
        base_image="image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/tensorflow"
        )
def train():
    print ('Cleaning data')

@dsl.component(
        base_image="image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/tensorflow"
        )
def infer():
    print ('Infering')

@dsl.pipeline(name='boston')
def boston_pipeline():
    task1 = gather_data()
    clean_data(inp=task1.output)
    train()
    infer()


compiler.Compiler().compile(boston_pipeline, "pipeline.yaml")    
