import gdown
import os 


# Download the data from Google Drive
#def download_data(file_id, file_name):
#    gdown.download(file_id, file_name, quiet=False)

os.mkdir("./api/assets")

gdown.download(
    "https://drive.google.com/uc?id=1trxIxT7D9OWmvzZNj3ZPFuroQLs9zncO",
    "./api/assets/GBV-Model.bin",
)