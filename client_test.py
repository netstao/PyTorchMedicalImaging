import requests
import json
import io
import torch

im, cl, id, pos = torch.load('data/cls_val_example.pt')

meta = io.StringIO(json.dumps({'shape': list(im.shape)}))
data = io.BytesIO(bytearray(im.numpy()))
r = requests.post('http://127.0.0.1:5000/predict',
                  files={'meta': meta, 'blob' : data})
response = json.loads(r.content)

print("Model predicted probability of being maignant:", response['prob_malignant'])
