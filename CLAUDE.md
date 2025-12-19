# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

llamafile is a Mozilla project that distributes and runs LLMs as single-file executables. It combines llama.cpp with Cosmopolitan Libc to create portable executables that run on Windows, macOS, Linux, FreeBSD, OpenBSD, and NetBSD without installation.

## Build Commands

```bash
# First-time setup: Initialize submodules and apply patches
make setup

# Build everything (uses cosmocc toolchain, downloads automatically)
make -j8

# Install to system
sudo make install PREFIX=/usr/local

# Clean build artifacts
make clean

# Full clean including toolchain
make distclean

# Run tests
make check
```

The build uses the Cosmopolitan Libc toolchain (`cosmocc`) which is downloaded automatically to `.cosmocc/`. Output goes to `o//` directory by default.

## Build System Architecture

- Uses GNU Make with custom rules in `build/config.mk` and `build/rules.mk`
- Each package has a `BUILD.mk` file defining targets
- Build outputs to `o/$(MODE)/` (default `o//`)
- Supports `MODE` variable for different build configurations (`make MODE=opt` or `make m=opt`)

## Key Directories

- `llamafile/` - Core llamafile library and utilities (zipalign, tokenize, CUDA/Metal runtime GPU compilation)
- `llamafile/server/` - HTTP API server (`llamafiler`) with OpenAI-compatible endpoints
- `llama.cpp/` - Git submodule with llamafile-specific patches applied
- `llama.cpp.patches/` - Patches and additional files for llama.cpp submodule
- `whisper.cpp/` and `whisper.cpp.patches/` - Whisper speech recognition (submodule + patches)
- `stable-diffusion.cpp/` and `stable-diffusion.cpp.patches/` - Image generation (submodule + patches)
- `localscore/` - LLM benchmarking tool
- `third_party/` - Dependencies (mbedtls, sqlite, stb, double-conversion)

## Submodule Patch System

Dependencies (llama.cpp, whisper.cpp, stable-diffusion.cpp) are managed as git submodules with llamafile-specific patches. The `make setup` command:
1. Initializes submodules
2. Copies llamafile-specific files from `*.patches/llamafile-files/`
3. Applies `.patch` files from `*.patches/patches/`

Patches modify submodule working directories without committing. To reset a submodule:
```bash
cd llama.cpp && git reset --hard && git clean -fdx
```

## Key Executables Built

- `o//llama.cpp/main/main` → `llamafile` (main CLI)
- `o//llamafile/server/main` → `llamafiler` (HTTP server)
- `o//llamafile/zipalign` → `zipalign` (embed weights into executables)
- `o//whisper.cpp/main` → `whisperfile` (speech recognition)
- `o//stable-diffusion.cpp/main` → `sdfile` (image generation)
- `o//localscore/localscore` → `localscore` (benchmarking)

## Architecture Notes

- **Portable executables**: Single binary runs on 6 OSes and 2 architectures (AMD64/ARM64)
- **Runtime GPU compilation**: CUDA and Metal support compiled at runtime using host compiler (`nvcc` or Xcode)
- **ZIP embedding**: Model weights embedded using page-aligned ZIP format for efficient mmap()
- **Microarchitecture dispatch**: CPU-specific optimizations (SSE, AVX, AVX2, AVX-512) selected at runtime

## Testing

Tests use the pattern `*_test.cpp` and are run via Make rules:
```bash
# Run all tests
make -j8  # Tests run as part of the build

# Individual test files become executables, e.g.:
o//llamafile/json_test
o//llamafile/pool_test
```

Test targets end with `.runs` suffix (e.g., `o/$(MODE)/llamafile/json_test.runs`).

## Creating llamafiles

To create a distributable llamafile with embedded weights:
```bash
# Copy base executable
cp o//llama.cpp/main/main my.llamafile

# Create .args file with default arguments
echo -e "-m\nmodel.gguf\n..." > .args

# Embed weights (must be page-aligned for GPU support)
o//llamafile/zipalign -j0 my.llamafile model.gguf .args
```

## Code Style

- C/C++ with GNU Make build system
- Uses Cosmopolitan Libc APIs
- Tab indentation in Makefiles
- Files use vi modelines for editor configuration
