import requests
import json

headers = {'Content-Type': 'application/json'}
url    = 'http://localhost:3300/'

def send_api(name, session=1, data=None):

    if data is None:
        res = requests.post('{}/{}/{}'.format(url, name, session),
                            headers=headers)
    else:
        res = requests.post('{}/{}/{}'.format(url, name, session),
                            headers=headers, data=data)

    
