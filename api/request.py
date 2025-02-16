import requests
import base64

with open("./data/sample.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

response = requests.post("http://localhost:8000/predict", json={"image": encoded_string})

print(response.json())