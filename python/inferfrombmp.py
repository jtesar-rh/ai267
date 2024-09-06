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

reqdata = '{ "id": "req", "inputs": [ { "name": "bitmap", "shape": [1,28,28], "datatype": "FP32", "data": ' + np.array2string(p.flatten()/255,max_line_width=1000000,floatmode="fixed",separator=",",precision=8) + '} ], "outputs": [ { "name": "probabilities" } ] }'

jsondata = json.loads(reqdata)
authheaders = {"Authorization": "Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6IlRjYmJ6ajdiR3k0bEZIWWZxSWhoRk12ZXFhN1dVLURLaS1LdkIwUGQyRkEifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJ0ZXN0MiIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VjcmV0Lm5hbWUiOiJkZWZhdWx0LW5hbWUtZmFzaGlvbi1zYSIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50Lm5hbWUiOiJmYXNoaW9uLXNhIiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZXJ2aWNlLWFjY291bnQudWlkIjoiOGFmYzU3ZTgtZjE1ZC00ZWZmLThjY2UtMjA0NWUyYTkzZDYzIiwic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OnRlc3QyOmZhc2hpb24tc2EifQ.VjHXDl_1Lcpf3TeoAq0394Q6axCoZAOpF_7kzzKf-Ee0wNjsiatdxXytEuHWklMsQ8XB0D0bpHPw8lYvxTMfBkZoqbmXvjSfsMY5JcWh5_cn2xHe7rVyoxi_C95d093UJ7rfTNkXxRsWAywm7THBNHQytscfFU7STDsbokNMClmYVQg0ZUjlDho7yZYuwBFZXGIjOgWVsXZLZzv0Hpfit4KcjVnoCji9JKf2Y6qNtZIN1h1zhgZFxiNJQKtnQrGmsfy_ZiSz5nwxjY3nC2b__xVw2ehO6_NPeat31BRtVfPmLiJzMrSPtJ-wZjy-QckDr5dksnCjLErMjswwjF5svnJeMgNpRYsHbzvM6lsj2JB3x35sZ3rdP3dr-I-cGvg2d__UfXFXyr5j4Hy2dzWwgjFT7g0Mj0ueRJSWn6pOMiMr6HJ-_5yVYTbr7sBo8RzUJjyQ1FIbEF-Rr1KJpguUDN14Lqv9Yzn0UvNiOxl9erxr1Wn07z3hGYxrxu0Ge-6wn0lgTBcwYNs9YiUUdO73U1tIyzpreNQ9ZF-FgebCYgm6l9XhyVfZF4bz6HUzXJiPyiZK204mRyAspHhUodQwWOvio59wturq_5R5pvp80V3PE3f05me0Ja4YY8DWGOfAUsNI8PuDE6TjS0zD_ih_e1oWk1yk7AWY0nmS-1gZvGo"}

req = requests.post('https://fashion-test2.apps.ocp4.example.com/v2/models/fashion/infer',json=jsondata,headers=authheaders)

response=(json.loads(req.text))["outputs"][0]["data"]
#resdata=response["outputs"][0]["data"]
#print (response["outputs"][0]["data"])
print (json.dumps(response,indent=4))
number = np.argmax(response)
print("The number on " + sys.argv[1] + " is number " + str(number) + ".\n")
