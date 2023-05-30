import requests
id_client = 100002
# Set the API URL
url = 'http://localhost:4000/predict/'+str(id_client)

# Send a GET request to the API
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Get the JSON response
    result = response.json()
    # Extract the predicted probability
    prediction = result['predictions']
    print('Predicted probability:', prediction)
else:
    print('Request failed with status code:', response.status_code)
