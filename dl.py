import kagglehub

# Download latest version
path = kagglehub.dataset_download("noahbadoa/plantnet-300k-images")

print("Path to dataset files:", path)
