#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += STABLE_DIFFUSION_CPP

# Metal source files to embed in the executable (for runtime compilation on macOS)
# These are extracted at runtime and compiled into ggml-metal.dylib
SD_METAL_SOURCES := \
	o/$(MODE)/llama.cpp/ggml/src/ggml.c.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-alloc.c.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-backend.cpp.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-quants.c.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-threading.cpp.zip.o \
	o/$(MODE)/llama.cpp/ggml/include/ggml.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/include/gguf.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/include/ggml-cpu.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/include/ggml-alloc.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/include/ggml-backend.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/include/ggml-metal.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-impl.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-common.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-quants.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-threading.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-backend-impl.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-cpu/ggml-cpu-impl.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal.cpp.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.cpp.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal-common.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal-common.cpp.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp.zip.o

STABLE_DIFFUSION_CPP_FILES := $(wildcard stable-diffusion.cpp/*.*)
STABLE_DIFFUSION_CPP_HDRS = $(filter %.h,$(STABLE_DIFFUSION_CPP_FILES))	\
			    $(filter %.hpp,$(STABLE_DIFFUSION_CPP_FILES))
STABLE_DIFFUSION_CPP_INCS = $(filter %.inc,$(STABLE_DIFFUSION_CPP_FILES))
STABLE_DIFFUSION_CPP_SRCS_C = $(filter %.c,$(STABLE_DIFFUSION_CPP_FILES))
STABLE_DIFFUSION_CPP_SRCS_CPP = $(filter-out stable-diffusion.cpp/main.cpp stable-diffusion.cpp/server.cpp,$(filter %.cpp,$(STABLE_DIFFUSION_CPP_FILES)))
STABLE_DIFFUSION_CPP_SRCS = $(STABLE_DIFFUSION_CPP_SRCS_C) $(STABLE_DIFFUSION_CPP_SRCS_CPP)

# Also compile zip.c from thirdparty
STABLE_DIFFUSION_CPP_THIRDPARTY_SRCS_C = stable-diffusion.cpp/thirdparty/zip.c

STABLE_DIFFUSION_CPP_OBJS =						\
	$(STABLE_DIFFUSION_CPP_SRCS_C:%.c=o/$(MODE)/%.o)		\
	$(STABLE_DIFFUSION_CPP_SRCS_CPP:%.cpp=o/$(MODE)/%.o)	\
	$(STABLE_DIFFUSION_CPP_THIRDPARTY_SRCS_C:%.c=o/$(MODE)/%.o)

o/$(MODE)/stable-diffusion.cpp/stable-diffusion.cpp.a: $(STABLE_DIFFUSION_CPP_OBJS)

$(STABLE_DIFFUSION_CPP_OBJS): private					\
		CCFLAGS +=						\
			-DGGML_MULTIPLATFORM

$(STABLE_DIFFUSION_CPP_OBJS): private					\
		CXXFLAGS +=						\
			-frtti						\
			-Wno-deprecated-declarations

o/$(MODE)/stable-diffusion.cpp/main:					\
		o/$(MODE)/stable-diffusion.cpp/main.o			\
		o/$(MODE)/stable-diffusion.cpp/server.o			\
		o/$(MODE)/stable-diffusion.cpp/stable-diffusion.cpp.a	\
		o/$(MODE)/llama.cpp/llama.cpp.a				\
		o/$(MODE)/llamafile/llamafile.o				\
		o/$(MODE)/llamafile/metal.o				\
		o/$(MODE)/llamafile/zip.o				\
		$(SD_METAL_SOURCES)					\
		o/$(MODE)/third_party/stb/stb.a

$(STABLE_DIFFUSION_CPP_OBJS): stable-diffusion.cpp/BUILD.mk

.PHONY: o/$(MODE)/stable-diffusion.cpp
o/$(MODE)/stable-diffusion.cpp:						\
		o/$(MODE)/stable-diffusion.cpp/main
