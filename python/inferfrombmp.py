#!/bin/python

from PIL import Image
import numpy as np
import sys
import os
import json
import requests


if len(sys.argv) == 1:
    print("Enter bitmap number")
    exit(-1)


im = Image.open(sys.argv[1])
p = np.array(im)

reqdata = '{ "id": "req", "inputs": [ { "name": "bitmap", "shape": [1,28,28], "datatype": "FP64", "data": ' + np.array2string(p.flatten()/255,max_line_width=1000000,floatmode="fixed",separator=",",precision=8) + '} ], "outputs": [ { "name": "probabilities" } ] }'

jsondata = json.loads(reqdata)
authheaders = {"Authorization": "Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6InJCb1BhM0NYQk5BZ0FXSVhfWjdiU1VCS0JQRXBFZ1JXN0JhVTAwVFhoSW8ifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJtb2RlbC1zZXJ2aW5nIiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZWNyZXQubmFtZSI6ImRlZmF1bHQtbmFtZS1udW1iZXJzLXNhIiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZXJ2aWNlLWFjY291bnQubmFtZSI6Im51bWJlcnMtc2EiLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC51aWQiOiIwODhkNmQ2ZC1iYWFjLTQ3YzItOGQyYi1iMmM2ZjYwM2ZmYzMiLCJzdWIiOiJzeXN0ZW06c2VydmljZWFjY291bnQ6bW9kZWwtc2VydmluZzpudW1iZXJzLXNhIn0.CP3NSCr45Xl8Y9FWAIky3WX8rKqWNZeq33mm9Eh8xxq8aG2hcCFVoyw5tQDQurHGFT0PwVI1MjMmwXd-1OcKjttx7F8jGwwMnmWmT14WbcGZTE66mBQoRJ9Ewod_nnJs9pFlJsyV01OcPtSOj68r55NKNsxLjngAnefpXT2zTFq7ZZSnZBHTuFJReRHoDMIKtRj3xE_v-du50YL84VtXb3D7mH_RaqsVdzB8JlJoF1yhkl17KsaGqVFcH9SorfW8LW4LcMhg0or3kKnVs032rWI-yU1e3LMaoq0bSik9d1zAGWIsF_wWtWTmovQjzpdiLSugqjVP6Gr9MuusA6-BSqEsemK0jqGAeg9ZJa4Rl5JASnMFLbwa5G-g9o-Nm9EXgRRlduLK5BX8dOPC8N5m7hzb9PoGSlop3PA8i1a7ZJ90U1CY04sTOnBnczYJwBskqUrRZYO_wR3_0IpScZ1AsYc9QntIkuzER7IxKRWpDxaYPqG9YPl9kANfvhR1ZpqG5n0GuEzoYXNksoaGsiQK0vAkQ9Vl96nsx-HZcwsidkrHyLpZD9_VhjXUrYrGBvQsoHWrEriTt5ThioZl1-djxGcqVFgev-V3zEMb67csgHUyUz7d0gOR9k1sXXZXkCB1KgGdUl8uiPzPUCn8fd6jVFdtw5yd_oKwu8SrBwBZ10k"}

req = requests.post('https://numbers-model-serving.apps.ocp4.example.com/v2/models/numbers/infer',json=jsondata,headers=authheaders)

response=(json.loads(req.text))["outputs"][0]["data"]
#resdata=response["outputs"][0]["data"]
#print (response["outputs"][0]["data"])
print (json.dumps(response,indent=4))
number = np.argmax(response)
print("The number on " + sys.argv[1] + " is number " + str(number) + ".\n")
