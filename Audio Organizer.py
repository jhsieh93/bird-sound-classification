# organize_kaggle_data.py
import pandas as pd
import os
import shutil

# ===== UPDATE THESE PATHS =====
METADATA_FILE = r'C:\Users\Jordan\Downloads\archive\wavfiles\bird_songs_metadata.csv'  # Your CSV file name (might be different)
AUDIO_SOURCE_DIR = r'C:\Users\Jordan\Downloads\archive\wavfiles'  # Where your .wav files are (current directory)
OUTPUT_DIR = r'C:\Users\Jordan\Desktop\Birds'  # Where to create organized folders
# ==============================

print("Reading metadata...")
df = pd.read_csv(METADATA_FILE)

print(f"Found {len(df)} audio files in metadata")
print(f"Species in dataset: {df['name'].nunique()} unique species")

# Show first few species
print("\nSample species:")
for species in df['name'].unique()[:5]:
    count = len(df[df['name'] == species])
    print(f"  {species}: {count} recordings")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Organize files by species
print(f"\nOrganizing files into {OUTPUT_DIR}/...")
organized_count = 0
missing_count = 0

for idx, row in df.iterrows():
    species_name = row['name']
    filename = row['filename']

    # Create species folder (replace spaces and special chars)
    species_folder = species_name.replace('/', '_').replace('\\', '_')
    species_path = os.path.join(OUTPUT_DIR, species_folder)
    os.makedirs(species_path, exist_ok=True)

    # Source and destination paths
    source_file = os.path.join(AUDIO_SOURCE_DIR, filename)
    dest_file = os.path.join(species_path, filename)

    # Copy file if it exists
    if os.path.exists(source_file):
        shutil.copy2(source_file, dest_file)
        organized_count += 1

        if organized_count % 100 == 0:
            print(f"  Organized {organized_count} files...")
    else:
        missing_count += 1
        if missing_count <= 5:  # Only show first few missing
            print(f"  WARNING: File not found: {filename}")

print(f"\n✓ Done!")
print(f"  Organized: {organized_count} files")
print(f"  Missing: {missing_count} files")
print(f"  Output location: {OUTPUT_DIR}/")

# Summary
species_folders = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
print(f"\nCreated {len(species_folders)} species folders")
print("\nYou can now use:")
print(f"  AUDIO_DIR = '{OUTPUT_DIR}'")
print("in your bird_classifier.py script!")


