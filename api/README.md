# GenderBasedViolence API

![api](./images/image.png)

## Project setup

install the dependencies for this project by running the following commands in your terminal:
```
 pip install -r requirements.txt
```

1. donwload the model from Google Drive.
```
python download_model.py
```

2. Launch the service with the following command:
```
cd api/api

uvicorn main:app
```

## Posting requests locally
When the service is running, try
```
127.0.0.1:8000/docs#
```
or 
```
curl
```

## Deployment with Docker
1. Build the Docker image
```
docker build --file Dockerfile --tag gbv-api .
```

2. Running the Docker image
```
docker run -p 8000:8000 gbv-api
```

