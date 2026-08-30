import pytest
import os
import urllib.request
from bionemo.rldif import RLDIF, RLDIFConfig, RLDIFDataset, RLDIF_Generator

# URL of the PDB file
PDB_URL = "https://files.rcsb.org/download/2C9K.pdb"
PDB_FILE_PATH = "2C9K.pdb"

# Helper function to calculate the level of overlap
def calculate_overlap(seq1, seq2):
    assert len(seq1) == len(seq2), "Sequences must be of the same length"
    overlap = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return overlap / len(seq1)

# Helper function to download PDB file
def download_pdb_file(url, file_path):
    if not os.path.exists(file_path):
        print(f"Downloading PDB file from {url}")
        urllib.request.urlretrieve(url, file_path)
        print(f"Saved PDB file to {file_path}")
    else:
        print(f"PDB file already exists at {file_path}")

# Test function
@pytest.mark.parametrize("pdb_url, pdb_file_path", [
    (PDB_URL, PDB_FILE_PATH)
])
def test_rldif_overlap(pdb_url, pdb_file_path):
    # Download the PDB file
    download_pdb_file(pdb_url, pdb_file_path)

    # Set up the PDB file paths
    pdb_file_paths = [pdb_file_path]

    # Initialize the model and other necessary classes
    config = RLDIFConfig()
    model = RLDIF(config).cuda()
    model.initialize()

    # Create the dataloader
    dataloader = RLDIFDataset(pdb_file_paths).return_dataloader(model)

    # Generate results
    result = RLDIF_Generator(model, dataloader, num_samples=4)

    # Print result for debugging
    print(result)

    # Extract real and predicted sequences - assume the DataFrame has columns 'real' and 'pred'
    real_sequence = result['real'].iloc[0]
    predicted_sequence = result['pred'].iloc[0]

    # Calculate the overlap
    overlap = calculate_overlap(real_sequence, predicted_sequence)
    print(f"Overlap: {overlap * 100}%")

    # Assert that the overlap is over 20%
    assert overlap > 0.20, f"Overlap is too low: {overlap * 100}%"

if __name__ == "__main__":
    pytest.main()