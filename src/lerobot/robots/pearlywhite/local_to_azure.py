from lerobot.datasets.lerobot_dataset import LeRobotDataset
import adlfs
from huggingface_hub import login, upload_folder, create_repo, create_tag, snapshot_download
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

'''Save dataset to Azure Blob'''
output_dir = "" # local folder name
sas_token = "" # token in Azure Blob Storage
azure_folder_name = "" # Your folder name in azure 

# Get dataset from huggingface. Output seen in Data storage -> containers
abfs = adlfs.AzureBlobFileSystem(account_name="pearlywhite", sas_token=sas_token)
abfs.mkdir(azure_folder_name)
print(abfs.ls(""))

# Save folder to azure blob
abfs.put(output_dir, "my-private-datasets", recursive=True)



'''Retrieve dataset from the cloud'''

# Download whole folder to locally save in the azure_data variable./hello6_from_azure
azure_data = "hello10_from_azure"
abfs.get(
    f"my-private-datasets/{output_dir}",  # remote path in Azure Blob
    azure_data, # local path to save
    recursive=True
)

# Push back the azure data to huggingFace
create_repo(
    f"rdteteam/{azure_data}", 
    repo_type="dataset", 
    private=True
)
upload_folder( 
    folder_path=azure_data, # your local folder
    repo_id=f"rdteteam/{azure_data}", # your repo on Hugging Face
    repo_type="dataset" # since it’s a dataset
)

# Download locally to view the file
snapshot_download(
    repo_id=f"rdteteam/{azure_data}",
    repo_type="dataset",
    local_dir=f"../.cache/huggingface/lerobot/rdteteam/{azure_data}",
    local_dir_use_symlinks=False
)
