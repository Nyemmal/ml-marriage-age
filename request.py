import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'gender':0, 'religion':5, 'caste':4, 'mother_tongue':4, 'country':6, 'height_cms':175  })

print(r.json())