import os
from pathlib import Path
from syftbox.lib import Client, SyftPermission

API_NAME = os.getenv("API_NAME")

def network_participants(datasite_path: Path) -> list[str]:
    """
    Network Participants Discovery:
    Retrieves a list of user directories (participants) in a given datasite path. 
    This function scans the network for all available peers by looking at directories in the datasite path.
    
    By looking for "api_data / API_NAME" only those from the specific app will be considered.
    
    """

    entries = sorted(os.listdir(datasite_path))
    users = []

    for entry in entries:
        if Path(datasite_path / entry / "api_data" / API_NAME).is_dir():
            users.append(entry)

    return users


def create_shared_folder(path: Path, client: Client, participants: list) -> Path:
    """
    Create a shared folder accessible to participants only with the computations
    """

    shared_datapath: Path = path / "api_data" / API_NAME
    os.makedirs(shared_datapath, exist_ok=True)

    # Set the default permissions
    permissions = SyftPermission.datasite_default(email=client.email)
    for participant in participants: # set read permission to participants
        permissions.read.append(participant)
    permissions.save(shared_datapath)  # update the ._syftperm

    return shared_datapath