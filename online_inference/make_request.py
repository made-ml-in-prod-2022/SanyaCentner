import numpy as np
import pandas as pd
import requests

if __name__ == "__main__":
    data = pd.read_csv("../data_csv/test_data.csv")
    del data['Unnamed: 0']
    request_features = list(data.columns)
    for i in range(100):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]
        print("Request: ", request_data)
        response = requests.get(
            "http://0.0.0.0:8000/predict",
            json={"data": [request_data], "features": request_features},
        )
        print("Response: ", response.status_code)
        print(response.json())
