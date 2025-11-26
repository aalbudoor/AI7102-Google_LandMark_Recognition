#!/bin/bash
set -e  # stop on errors

# Define paths
SRC="/home/abdulla.alshehhi/Desktop/r/AI7102-Google_LandMark_Recognition/data/images"
DST="/home/abdulla.alshehhi/Desktop/r/AI7102-Google_LandMark_Recognition/data/flat_images"

# Create destination directory if it doesn’t exist
mkdir -p "$DST"

echo " Flattening all .jpg images from:"
echo "    $SRC"
echo "➡️  to:"
echo "    $DST"
echo ""

# Counter for renamed duplicates
counter=0

# Find and copy all .jpg/.JPG files
find "$SRC" -type f \( -iname "*.jpg" \) | while read -r file; do
    filename=$(basename "$file")
    base="${filename%.*}"
    ext="${filename##*.}"
    
    # If duplicate exists, append a counter
    if [[ -e "$DST/$filename" ]]; then
        counter=$((counter + 1))
        newname="${base}_${counter}.${ext}"
        cp "$file" "$DST/$newname"
    else
        cp "$file" "$DST/"
    fi
done

echo "✅ Done! All .jpg files have been copied to: $DST"
