from lerobot.datasets.lerobot_dataset import LeRobotDataset
import adlfs
from huggingface_hub import login, upload_folder, create_repo, create_tag, snapshot_download
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

'''Save dataset to Azure Blob'''
output_dir = "my_output_hello3"
# lerobot = LeRobotDataset(repo_id='rdteteam/hello3', root=output_dir)
# lerobot.pull_from_repo()

# # Get dataset from huggingface. Output seen in Data storage -> containers
abfs = adlfs.AzureBlobFileSystem(account_name="pearlywhite", sas_token="sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2025-10-03T14:52:54Z&st=2025-09-26T06:37:54Z&spr=https&sig=XUEpsNIFsZuOuoK%2FvSBooo1PEs3mI1vHLytPsZ4DQxo%3D")
# abfs.mkdir("my-private-datasets")
# print(abfs.ls(""))

# # Save folder to azure blob
# abfs.put(output_dir, "my-private-datasets", recursive=True)

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
