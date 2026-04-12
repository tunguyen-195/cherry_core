"""
Inspect VIVOS Repo files to find download path.
"""
from huggingface_hub import list_repo_files

try:
    print("Listing files in 'vivos' dataset repo...")
    files = list_repo_files(repo_id="vivos", repo_type="dataset")
    print("\nFiles found:")
    for f in files[:20]:  # Show first 20
        print(f" - {f}")
except Exception as e:
    print(f"Error: {e}")
