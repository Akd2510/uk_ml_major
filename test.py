import netCDF4 as nc
import numpy as np
import os
import glob

# Load the NetCDF file
data_dir = r"C:\Users\ABHIMANYU\OneDrive\Desktop\ml landslide project\data"
nc_files = glob.glob(os.path.join(data_dir, "*.nc"))

if not nc_files:
    raise FileNotFoundError(f"No .nc files found in {data_dir}")

file_path = nc_files[0]
print(f"Loading {os.path.basename(file_path)}...")

ds = nc.Dataset(file_path, 'r')

print("=" * 60)
print("FILE INFO")
print("=" * 60)
print(f"File format : {ds.file_format}")

# Global attributes
print("\n--- Global Attributes ---")
for attr in ds.ncattrs():
    print(f"  {attr}: {getattr(ds, attr)}")

# Dimensions
print("\n--- Dimensions ---")
for dim_name, dim in ds.dimensions.items():
    print(f"  {dim_name}: size = {len(dim)}")

# Variables
print("\n--- Variables ---")
for var_name, var in ds.variables.items():
    print(f"\n  Variable : {var_name}")
    print(f"  Shape    : {var.shape}")
    print(f"  Dtype    : {var.dtype}")
    print(f"  Dims     : {var.dimensions}")
    for attr in var.ncattrs():
        print(f"    {attr}: {getattr(var, attr)}")

# Print sample data for each variable
print("\n" + "=" * 60)
print("SAMPLE DATA")
print("=" * 60)
for var_name, var in ds.variables.items():
    data = var[:]
    print(f"\n  [{var_name}]")
    print(f"  Min   : {np.nanmin(data):.4f}")
    print(f"  Max   : {np.nanmax(data):.4f}")
    print(f"  Mean  : {np.nanmean(data):.4f}")
    print(f"  Sample values (first 5 flat): {data.flatten()[:5]}")

ds.close()
print("\nDone.")