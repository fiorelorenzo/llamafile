#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SD_DIR="$SCRIPT_DIR/../stable-diffusion.cpp"
PATCHES_DIR="$SCRIPT_DIR/patches"
LLAMAFILE_FILES_DIR="$SCRIPT_DIR/llamafile-files"

cd "$SD_DIR"

if [ -f "BUILD.mk" ]; then
    echo "Patches appear to be already applied. Skipping..."
    exit 0
fi

echo "Applying patches to stable-diffusion.cpp submodule..."

# Convert CRLF to LF for files that may have Windows line endings
# This ensures patches apply correctly
echo "Normalizing line endings..."
for f in latent-preview.h; do
    if [ -f "$f" ]; then
        sed -i '' $'s/\r$//' "$f" 2>/dev/null || sed -i 's/\r$//' "$f" 2>/dev/null || true
    fi
done

echo "Applying patch files..."
for patch_file in "$PATCHES_DIR"/*.patch; do
    if [ -f "$patch_file" ]; then
        echo "Applying $(basename "$patch_file")..."
        git apply --whitespace=nowarn "$patch_file" || patch -p1 --ignore-whitespace < "$patch_file"
    fi
done

# latent-preview.h has CRLF line endings that cause patch to fail
# Apply the fix manually after normalizing (was done above)
echo "Fixing latent-preview.h include path..."
if [ -f "latent-preview.h" ]; then
    sed -i '' 's|#include "ggml.h"|#include "llama.cpp/ggml/include/ggml.h"|g' latent-preview.h 2>/dev/null || \
    sed -i 's|#include "ggml.h"|#include "llama.cpp/ggml/include/ggml.h"|g' latent-preview.h
fi

echo "Copying llamafile-specific files..."
cp "$LLAMAFILE_FILES_DIR/BUILD.mk" .
cp "$LLAMAFILE_FILES_DIR/README.llamafile" .
cp "$LLAMAFILE_FILES_DIR/main.cpp" .
cp "$LLAMAFILE_FILES_DIR/server.cpp" .

echo "Removing unnecessary files and directories..."
rm -rf .github
rm -rf assets
rm -rf docs
rm -rf examples
rm -rf ggml
rm -rf models
rm -f .clang-format
rm -f .dockerignore
rm -f .gitignore
rm -f .gitmodules
rm -f CMakeLists.txt
rm -f Dockerfile
rm -f format-code.sh
rm -f README.md

echo "Cleaning thirdparty directory (keeping needed files)..."
# Keep httplib.h, zip.h, zip.c, miniz.h, and darts.h from thirdparty - these are needed
if [ -d "thirdparty" ]; then
    find thirdparty -type f ! -name 'httplib.h' ! -name 'zip.h' ! -name 'zip.c' ! -name 'miniz.h' ! -name 'darts.h' -delete
    rm -f thirdparty/.clang-format thirdparty/CMakeLists.txt thirdparty/README.md
fi

echo ""
echo "Patches applied successfully!"
echo "Note: These changes are not committed to the submodule."
echo "To reset the submodule to its clean state, run:"
echo "  cd stable-diffusion.cpp && git reset --hard && git clean -fd"
