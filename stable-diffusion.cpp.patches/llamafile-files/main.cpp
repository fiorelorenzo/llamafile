// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;tab-width:8;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>
// From llama.cpp/common/common.h - declared here to avoid heavy includes
int32_t cpu_get_num_math();
#include <cosmo.h>

#include "stable-diffusion.h"

#include "third_party/stb/stb_image.h"
#include "third_party/stb/stb_image_write.h"
#include "third_party/stb/stb_image_resize2.h"

#include "llamafile/llamafile.h"
#include "llama.cpp/ggml/include/ggml.h"

// Forward declaration for server
int sd_server_main(int argc, const char** argv);

static const char* rng_type_to_str[] = {
    "std_default",
    "cuda",
    "cpu",
};

static const char* sample_method_str[] = {
    "euler",
    "euler_a",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "ipndm",
    "ipndm_v",
    "lcm",
    "ddim_trailing",
    "tcd",
};

static const char* scheduler_str[] = {
    "discrete",
    "karras",
    "exponential",
    "ays",
    "gits",
    "sgm_uniform",
    "simple",
    "smoothstep",
    "lcm",
};

static const char* modes_str[] = {
    "txt2img",
    "img2img",
    "vid_gen",
    "convert",
};

enum SDMode {
    TXT2IMG,
    IMG2IMG,
    VID_GEN,
    CONVERT,
    MODE_COUNT
};

struct SDParams {
    int n_threads = -1;
    SDMode mode   = TXT2IMG;
    bool server_mode = false;

    // Model paths
    std::string model_path;              // Full model (combined)
    std::string diffusion_model_path;    // Standalone diffusion model
    std::string vae_path;
    std::string taesd_path;
    std::string esrgan_path;
    std::string controlnet_path;
    std::string embeddings_path;
    std::string photo_maker_path;
    std::string input_id_images_path;

    // Text encoder paths (for SD3/Flux/etc)
    std::string clip_l_path;
    std::string clip_g_path;
    std::string t5xxl_path;
    std::string llm_path;                // LLM text encoder (for z-image/qwen)
    std::string llm_vision_path;         // LLM vision (for z-image)

    sd_type_t wtype = SD_TYPE_COUNT;
    std::string output_path = "output.png";
    std::string input_path;
    std::string control_image_path;

    std::string prompt;
    std::string negative_prompt;
    float cfg_scale        = 7.0f;
    float guidance         = 3.5f;       // distilled guidance for Flux/etc
    float style_ratio      = 20.f;
    int clip_skip          = -1;
    int width              = 512;
    int height             = 512;
    int batch_count        = 1;

    sample_method_t sample_method = EULER_A_SAMPLE_METHOD;
    scheduler_t scheduler         = DISCRETE_SCHEDULER;
    int sample_steps              = 20;
    float strength                = 0.75f;
    float control_strength        = 0.9f;
    rng_type_t rng_type           = CUDA_RNG;
    int64_t seed                  = 42;
    bool verbose                  = false;
    bool vae_tiling               = false;
    bool control_net_cpu          = false;
    bool normalize_input          = false;
    bool clip_on_cpu              = false;
    bool vae_on_cpu               = false;
    bool canny_preprocess         = false;
    bool color                    = false;
    int upscale_repeats           = 1;

    // Advanced options
    bool offload_to_cpu    = false;      // Offload params to CPU
    bool diffusion_fa      = false;      // Flash attention for diffusion

    // Server options
    std::string listen_ip = "127.0.0.1";
    int listen_port       = 7860;
};

void print_params(SDParams params) {
    printf("Option: \n");
    printf("    n_threads:         %d\n", params.n_threads);
    printf("    mode:              %s\n", modes_str[params.mode]);
    printf("    model_path:        %s\n", params.model_path.c_str());
    printf("    diffusion_model:   %s\n", params.diffusion_model_path.c_str());
    printf("    wtype:             %s\n", params.wtype < SD_TYPE_COUNT ? sd_type_name(params.wtype) : "unspecified");
    printf("    vae_path:          %s\n", params.vae_path.c_str());
    printf("    taesd_path:        %s\n", params.taesd_path.c_str());
    printf("    esrgan_path:       %s\n", params.esrgan_path.c_str());
    printf("    controlnet_path:   %s\n", params.controlnet_path.c_str());
    printf("    embeddings_path:   %s\n", params.embeddings_path.c_str());
    printf("    clip_l_path:       %s\n", params.clip_l_path.c_str());
    printf("    clip_g_path:       %s\n", params.clip_g_path.c_str());
    printf("    t5xxl_path:        %s\n", params.t5xxl_path.c_str());
    printf("    llm_path:          %s\n", params.llm_path.c_str());
    printf("    llm_vision_path:   %s\n", params.llm_vision_path.c_str());
    printf("    photo_maker_path:  %s\n", params.photo_maker_path.c_str());
    printf("    input_id_images:   %s\n", params.input_id_images_path.c_str());
    printf("    style ratio:       %.2f\n", params.style_ratio);
    printf("    normalize input:   %s\n", params.normalize_input ? "true" : "false");
    printf("    output_path:       %s\n", params.output_path.c_str());
    printf("    init_img:          %s\n", params.input_path.c_str());
    printf("    control_image:     %s\n", params.control_image_path.c_str());
    printf("    offload_to_cpu:    %s\n", params.offload_to_cpu ? "true" : "false");
    printf("    clip on cpu:       %s\n", params.clip_on_cpu ? "true" : "false");
    printf("    controlnet cpu:    %s\n", params.control_net_cpu ? "true" : "false");
    printf("    vae decoder on cpu:%s\n", params.vae_on_cpu ? "true" : "false");
    printf("    diffusion_fa:      %s\n", params.diffusion_fa ? "true" : "false");
    printf("    strength(control): %.2f\n", params.control_strength);
    printf("    prompt:            %s\n", params.prompt.c_str());
    printf("    negative_prompt:   %s\n", params.negative_prompt.c_str());
    printf("    cfg_scale:         %.2f\n", params.cfg_scale);
    printf("    guidance:          %.2f\n", params.guidance);
    printf("    clip_skip:         %d\n", params.clip_skip);
    printf("    width:             %d\n", params.width);
    printf("    height:            %d\n", params.height);
    printf("    sample_method:     %s\n", sample_method_str[params.sample_method]);
    printf("    scheduler:         %s\n", scheduler_str[params.scheduler]);
    printf("    sample_steps:      %d\n", params.sample_steps);
    printf("    strength(img2img): %.2f\n", params.strength);
    printf("    rng:               %s\n", rng_type_to_str[params.rng_type]);
    printf("    seed:              %ld\n", params.seed);
    printf("    batch_count:       %d\n", params.batch_count);
    printf("    vae_tiling:        %s\n", params.vae_tiling ? "true" : "false");
    printf("    upscale_repeats:   %d\n", params.upscale_repeats);
}

void print_usage(int argc, const char* argv[]) {
    printf("usage: %s [arguments]\n", argv[0]);
    printf("\n");
    printf("arguments:\n");
    printf("  -h, --help                         show this help message and exit\n");
    printf("  --server                           run in server mode (HTTP API)\n");
    printf("  --host IP                          server listen IP (default: 127.0.0.1)\n");
    printf("  --port PORT                        server listen port (default: 7860)\n");
    printf("  -M, --mode [MODE]                  run mode (txt2img, img2img, convert, default: txt2img)\n");
    printf("  -t, --threads N                    number of threads to use (default: -1 = auto)\n");
    printf("  -m, --model [MODEL]                path to full model\n");
    printf("  --diffusion-model [PATH]           path to standalone diffusion model\n");
    printf("  --vae [VAE]                        path to vae\n");
    printf("  --taesd [PATH]                     path to taesd (fast low quality decoding)\n");
    printf("  --clip_l [PATH]                    path to clip-l text encoder\n");
    printf("  --clip_g [PATH]                    path to clip-g text encoder\n");
    printf("  --t5xxl [PATH]                     path to t5xxl text encoder\n");
    printf("  --llm [PATH]                       path to LLM text encoder (for z-image/qwen)\n");
    printf("  --llm_vision [PATH]                path to LLM vision encoder\n");
    printf("  --control-net [PATH]               path to control net model\n");
    printf("  --embd-dir [PATH]                  path to embeddings\n");
    printf("  --upscale-model [PATH]             path to esrgan upscaler model\n");
    printf("  --upscale-repeats N                run upscaler N times (default: 1)\n");
    printf("  --type [TYPE]                      weight type (f32, f16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0, ...)\n");
    printf("  -i, --init-img [IMAGE]             input image for img2img\n");
    printf("  --control-image [IMAGE]            control net condition image\n");
    printf("  -o, --output OUTPUT                output image path (default: output.png)\n");
    printf("  -p, --prompt [PROMPT]              the prompt to render\n");
    printf("  -n, --negative-prompt PROMPT       negative prompt (default: \"\")\n");
    printf("  --cfg-scale SCALE                  unconditional guidance scale (default: 7.0)\n");
    printf("  --guidance SCALE                   distilled guidance for Flux/etc (default: 3.5)\n");
    printf("  --strength STRENGTH                img2img strength (default: 0.75)\n");
    printf("  --control-strength STRENGTH        control net strength (default: 0.9)\n");
    printf("  -H, --height H                     image height (default: 512)\n");
    printf("  -W, --width W                      image width (default: 512)\n");
    printf("  --sampling-method METHOD           sampling method (default: euler_a)\n");
    printf("  --scheduler SCHEDULER              scheduler (default: discrete)\n");
    printf("  --steps STEPS                      number of sample steps (default: 20)\n");
    printf("  --rng {std_default, cuda, cpu}     RNG type (default: cuda)\n");
    printf("  -s, --seed SEED                    RNG seed (default: 42, random if < 0)\n");
    printf("  -b, --batch-count COUNT            number of images to generate\n");
    printf("  --clip-skip N                      CLIP skip layers (default: -1 = auto)\n");
    printf("  --vae-tiling                       process VAE in tiles (saves memory)\n");
    printf("  --offload-to-cpu                   offload weights to CPU to save VRAM\n");
    printf("  --diffusion-fa                     use flash attention in diffusion model\n");
    printf("  --clip-on-cpu                      keep CLIP on CPU\n");
    printf("  --vae-on-cpu                       keep VAE on CPU\n");
    printf("  --control-net-cpu                  keep controlnet on CPU\n");
    printf("  --canny                            apply canny edge detection\n");
    printf("  --color                            colored log output\n");
    printf("  -v, --verbose                      print extra info\n");
    printf("\n");
    printf("llamafile options:\n");
    printf("  --gpu GPU                          GPU to use (e.g., auto, nvidia, amd, apple, disable)\n");
    printf("  --fast                             use faster but less accurate math\n");
    printf("  --precise                          use more accurate math\n");
    printf("  --nocompile                        disable runtime GPU compilation\n");
    printf("  --recompile                        force GPU recompilation\n");
    printf("  --tinyblas                         use tinyblas\n");
    printf("  --trap                             enable crash trapping\n");
    printf("  --unsecure                         disable security features\n");
}

void parse_args(int argc, const char** argv, SDParams& params) {
    bool invalid_arg = false;
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        // llamafile options
        if (arg == "--fast") {
            FLAG_fast = true;
        } else if (arg == "--precise") {
            FLAG_precise = true;
        } else if (arg == "--trace") {
            FLAG_trace = true;
        } else if (arg == "--trap") {
            FLAG_trap = true;
            FLAG_unsecure = true;
        } else if (arg == "--unsecure") {
            FLAG_unsecure = true;
        } else if (arg == "--nocompile") {
            FLAG_nocompile = true;
        } else if (arg == "--recompile") {
            FLAG_recompile = true;
        } else if (arg == "--tinyblas") {
            FLAG_tinyblas = true;
        } else if (arg == "--gpu") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            FLAG_gpu = llamafile_gpu_parse(argv[i]);
            if (FLAG_gpu == LLAMAFILE_GPU_ERROR) {
                fprintf(stderr, "error: invalid --gpu flag value: %s\n", argv[i]);
                exit(1);
            }
        // Server options
        } else if (arg == "--server") {
            params.server_mode = true;
        } else if (arg == "--host") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.listen_ip = argv[i];
        } else if (arg == "--port") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.listen_port = std::stoi(argv[i]);
        // Standard options
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.n_threads = std::stoi(argv[i]);
        } else if (arg == "-M" || arg == "--mode") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            const char* mode_selected = argv[i];
            int mode_found = -1;
            for (int d = 0; d < MODE_COUNT; d++) {
                if (!strcmp(mode_selected, modes_str[d])) {
                    mode_found = d;
                }
            }
            if (mode_found == -1) {
                fprintf(stderr, "error: invalid mode %s\n", mode_selected);
                exit(1);
            }
            params.mode = (SDMode)mode_found;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.model_path = argv[i];
        } else if (arg == "--vae") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.vae_path = argv[i];
        } else if (arg == "--taesd" || arg == "--tae") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.taesd_path = argv[i];
        } else if (arg == "--diffusion-model") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.diffusion_model_path = argv[i];
        } else if (arg == "--clip_l") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.clip_l_path = argv[i];
        } else if (arg == "--clip_g") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.clip_g_path = argv[i];
        } else if (arg == "--t5xxl") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.t5xxl_path = argv[i];
        } else if (arg == "--llm" || arg == "--qwen2vl") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.llm_path = argv[i];
        } else if (arg == "--llm_vision" || arg == "--qwen2vl_vision") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.llm_vision_path = argv[i];
        } else if (arg == "--control-net") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.controlnet_path = argv[i];
        } else if (arg == "--upscale-model") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.esrgan_path = argv[i];
        } else if (arg == "--embd-dir") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.embeddings_path = argv[i];
        } else if (arg == "--type") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.wtype = str_to_sd_type(argv[i]);
            if (params.wtype == SD_TYPE_COUNT) {
                fprintf(stderr, "error: invalid weight type %s\n", argv[i]);
                exit(1);
            }
        } else if (arg == "-i" || arg == "--init-img") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.input_path = argv[i];
        } else if (arg == "--control-image") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.control_image_path = argv[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.output_path = argv[i];
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.prompt = argv[i];
        } else if (arg == "--upscale-repeats") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.upscale_repeats = std::stoi(argv[i]);
        } else if (arg == "-n" || arg == "--negative-prompt") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.negative_prompt = argv[i];
        } else if (arg == "--cfg-scale") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.cfg_scale = std::stof(argv[i]);
        } else if (arg == "--guidance") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.guidance = std::stof(argv[i]);
        } else if (arg == "--strength") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.strength = std::stof(argv[i]);
        } else if (arg == "--control-strength") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.control_strength = std::stof(argv[i]);
        } else if (arg == "-H" || arg == "--height") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.height = std::stoi(argv[i]);
        } else if (arg == "-W" || arg == "--width") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.width = std::stoi(argv[i]);
        } else if (arg == "--steps") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.sample_steps = std::stoi(argv[i]);
        } else if (arg == "--clip-skip") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.clip_skip = std::stoi(argv[i]);
        } else if (arg == "--vae-tiling") {
            params.vae_tiling = true;
        } else if (arg == "--offload-to-cpu") {
            params.offload_to_cpu = true;
        } else if (arg == "--diffusion-fa") {
            params.diffusion_fa = true;
        } else if (arg == "--control-net-cpu") {
            params.control_net_cpu = true;
        } else if (arg == "--normalize-input") {
            params.normalize_input = true;
        } else if (arg == "--clip-on-cpu") {
            params.clip_on_cpu = true;
        } else if (arg == "--vae-on-cpu") {
            params.vae_on_cpu = true;
        } else if (arg == "--canny") {
            params.canny_preprocess = true;
        } else if (arg == "-b" || arg == "--batch-count") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.batch_count = std::stoi(argv[i]);
        } else if (arg == "--rng") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.rng_type = str_to_rng_type(argv[i]);
            if (params.rng_type == RNG_TYPE_COUNT) {
                fprintf(stderr, "error: invalid rng type %s\n", argv[i]);
                exit(1);
            }
        } else if (arg == "--scheduler") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.scheduler = str_to_scheduler(argv[i]);
            if (params.scheduler == SCHEDULER_COUNT) {
                fprintf(stderr, "error: invalid scheduler %s\n", argv[i]);
                exit(1);
            }
        } else if (arg == "-s" || arg == "--seed") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.seed = std::stoll(argv[i]);
        } else if (arg == "--sampling-method") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.sample_method = str_to_sample_method(argv[i]);
            if (params.sample_method == SAMPLE_METHOD_COUNT) {
                fprintf(stderr, "error: invalid sampling method %s\n", argv[i]);
                exit(1);
            }
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv);
            exit(0);
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else if (arg == "--color") {
            params.color = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv);
            exit(1);
        }
    }
    if (invalid_arg) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        print_usage(argc, argv);
        exit(1);
    }
    if (params.n_threads <= 0) {
        params.n_threads = cpu_get_num_math();
    }

    // Server mode has different requirements
    if (params.server_mode) {
        if (params.model_path.empty() && params.diffusion_model_path.empty()) {
            fprintf(stderr, "error: --model or --diffusion-model is required for server mode\n");
            exit(1);
        }
        FLAGS_READY = true;
        return;
    }

    if (params.mode != CONVERT && params.prompt.empty()) {
        fprintf(stderr, "error: prompt is required\n");
        print_usage(argc, argv);
        exit(1);
    }

    if (params.model_path.empty() && params.diffusion_model_path.empty()) {
        fprintf(stderr, "error: --model or --diffusion-model is required\n");
        print_usage(argc, argv);
        exit(1);
    }

    if (params.mode == IMG2IMG && params.input_path.empty()) {
        fprintf(stderr, "error: img2img mode requires --init-img\n");
        print_usage(argc, argv);
        exit(1);
    }

    if (params.width <= 0 || params.width % 64 != 0) {
        fprintf(stderr, "error: width must be a positive multiple of 64\n");
        exit(1);
    }

    if (params.height <= 0 || params.height % 64 != 0) {
        fprintf(stderr, "error: height must be a positive multiple of 64\n");
        exit(1);
    }

    if (params.sample_steps <= 0) {
        fprintf(stderr, "error: sample_steps must be > 0\n");
        exit(1);
    }

    if (params.strength < 0.f || params.strength > 1.f) {
        fprintf(stderr, "error: strength must be in [0.0, 1.0]\n");
        exit(1);
    }

    if (params.seed < 0) {
        srand((int)time(NULL));
        params.seed = rand();
    }

    if (params.mode == CONVERT && params.output_path == "output.png") {
        params.output_path = "output.gguf";
    }

    FLAGS_READY = true;
}

static std::string sd_basename(const std::string& path) {
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    pos = path.find_last_of('\\');
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    return path;
}

std::string get_image_params(SDParams params, int64_t seed) {
    std::string parameter_string = params.prompt + "\n";
    if (!params.negative_prompt.empty()) {
        parameter_string += "Negative prompt: " + params.negative_prompt + "\n";
    }
    parameter_string += "Steps: " + std::to_string(params.sample_steps) + ", ";
    parameter_string += "CFG scale: " + std::to_string(params.cfg_scale) + ", ";
    parameter_string += "Seed: " + std::to_string(seed) + ", ";
    parameter_string += "Size: " + std::to_string(params.width) + "x" + std::to_string(params.height) + ", ";
    parameter_string += "Model: " + sd_basename(params.model_path) + ", ";
    parameter_string += "Sampler: " + std::string(sample_method_str[params.sample_method]);
    parameter_string += ", Version: sdfile";
    return parameter_string;
}

void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    SDParams* params = (SDParams*)data;
    int tag_color;
    const char* level_str;
    FILE* out_stream = (level == SD_LOG_ERROR) ? stderr : stdout;

    if (!log || (!params->verbose && level <= SD_LOG_DEBUG)) {
        return;
    }

    switch (level) {
        case SD_LOG_DEBUG:
            tag_color = 37;
            level_str = "DEBUG";
            break;
        case SD_LOG_INFO:
            tag_color = 34;
            level_str = "INFO";
            break;
        case SD_LOG_WARN:
            tag_color = 35;
            level_str = "WARN";
            break;
        case SD_LOG_ERROR:
            tag_color = 31;
            level_str = "ERROR";
            break;
        default:
            tag_color = 33;
            level_str = "?????";
            break;
    }

    if (params->color) {
        fprintf(out_stream, "\033[%d;1m[%-5s]\033[0m ", tag_color, level_str);
    } else {
        fprintf(out_stream, "[%-5s] ", level_str);
    }
    fputs(log, out_stream);
    fflush(out_stream);
}

int main(int argc, const char* argv[]) {
    ShowCrashReports();

    SDParams params;
    parse_args(argc, argv, params);

    sd_set_log_callback(sd_log_cb, (void*)&params);

    // Server mode
    if (params.server_mode) {
        return sd_server_main(argc, argv);
    }

    if (params.verbose) {
        print_params(params);
        printf("%s", sd_get_system_info());
    }

    if (params.mode == CONVERT) {
        bool success = convert(params.model_path.c_str(),
                               params.vae_path.c_str(),
                               params.output_path.c_str(),
                               params.wtype,
                               nullptr);
        if (!success) {
            fprintf(stderr, "convert failed\n");
            return 1;
        }
        printf("convert success: %s\n", params.output_path.c_str());
        return 0;
    }

    // Load input image for img2img
    bool vae_decode_only = true;
    uint8_t* input_image_buffer = NULL;
    uint8_t* control_image_buffer = NULL;

    if (params.mode == IMG2IMG) {
        vae_decode_only = false;
        int c = 0, width = 0, height = 0;
        input_image_buffer = stbi_load(params.input_path.c_str(), &width, &height, &c, 3);
        if (!input_image_buffer) {
            fprintf(stderr, "error: failed to load image: %s\n", params.input_path.c_str());
            return 1;
        }
        if (c < 3) {
            fprintf(stderr, "error: input image must have >= 3 channels\n");
            free(input_image_buffer);
            return 1;
        }

        // Resize if needed
        if (width != params.width || height != params.height) {
            printf("resizing input image from %dx%d to %dx%d\n", width, height, params.width, params.height);
            uint8_t* resized = (uint8_t*)malloc(params.width * params.height * 3);
            stbir_resize(input_image_buffer, width, height, 0,
                         resized, params.width, params.height, 0,
                         STBIR_RGB, STBIR_TYPE_UINT8_SRGB, STBIR_EDGE_CLAMP,
                         STBIR_FILTER_BOX);
            free(input_image_buffer);
            input_image_buffer = resized;
        }
    }

    // Create context using new API
    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path = params.model_path.empty() ? nullptr : params.model_path.c_str();
    ctx_params.diffusion_model_path = params.diffusion_model_path.empty() ? nullptr : params.diffusion_model_path.c_str();
    ctx_params.vae_path = params.vae_path.empty() ? nullptr : params.vae_path.c_str();
    ctx_params.taesd_path = params.taesd_path.empty() ? nullptr : params.taesd_path.c_str();
    ctx_params.control_net_path = params.controlnet_path.empty() ? nullptr : params.controlnet_path.c_str();
    ctx_params.clip_l_path = params.clip_l_path.empty() ? nullptr : params.clip_l_path.c_str();
    ctx_params.clip_g_path = params.clip_g_path.empty() ? nullptr : params.clip_g_path.c_str();
    ctx_params.t5xxl_path = params.t5xxl_path.empty() ? nullptr : params.t5xxl_path.c_str();
    ctx_params.llm_path = params.llm_path.empty() ? nullptr : params.llm_path.c_str();
    ctx_params.llm_vision_path = params.llm_vision_path.empty() ? nullptr : params.llm_vision_path.c_str();
    ctx_params.vae_decode_only = vae_decode_only;
    ctx_params.n_threads = params.n_threads;
    ctx_params.wtype = params.wtype;
    ctx_params.rng_type = params.rng_type;
    ctx_params.offload_params_to_cpu = params.offload_to_cpu;
    ctx_params.keep_clip_on_cpu = params.clip_on_cpu;
    ctx_params.keep_control_net_on_cpu = params.control_net_cpu;
    ctx_params.keep_vae_on_cpu = params.vae_on_cpu;
    ctx_params.diffusion_flash_attn = params.diffusion_fa;

    sd_ctx_t* sd_ctx = new_sd_ctx(&ctx_params);
    if (!sd_ctx) {
        fprintf(stderr, "error: failed to create sd context\n");
        free(input_image_buffer);
        return 1;
    }

    // Load control image
    sd_image_t control_image = {0, 0, 0, nullptr};
    if (!params.controlnet_path.empty() && !params.control_image_path.empty()) {
        int c = 0;
        control_image_buffer = stbi_load(params.control_image_path.c_str(), (int*)&control_image.width, (int*)&control_image.height, &c, 3);
        if (!control_image_buffer) {
            fprintf(stderr, "error: failed to load control image: %s\n", params.control_image_path.c_str());
            free_sd_ctx(sd_ctx);
            free(input_image_buffer);
            return 1;
        }
        control_image.channel = 3;
        control_image.data = control_image_buffer;

        if (params.canny_preprocess) {
            preprocess_canny(control_image, 0.08f, 0.08f, 0.8f, 1.0f, false);
        }
    }

    // Set up generation parameters
    sd_img_gen_params_t gen_params;
    sd_img_gen_params_init(&gen_params);
    gen_params.prompt = params.prompt.c_str();
    gen_params.negative_prompt = params.negative_prompt.c_str();
    gen_params.clip_skip = params.clip_skip;
    gen_params.width = params.width;
    gen_params.height = params.height;
    gen_params.seed = params.seed;
    gen_params.batch_count = params.batch_count;
    gen_params.strength = params.strength;
    gen_params.control_image = control_image;
    gen_params.control_strength = params.control_strength;

    // Sample parameters
    sd_sample_params_init(&gen_params.sample_params);
    gen_params.sample_params.sample_method = params.sample_method;
    gen_params.sample_params.scheduler = params.scheduler;
    gen_params.sample_params.sample_steps = params.sample_steps;
    gen_params.sample_params.guidance.txt_cfg = params.cfg_scale;
    gen_params.sample_params.guidance.distilled_guidance = params.guidance;

    // VAE tiling
    gen_params.vae_tiling_params.enabled = params.vae_tiling;

    // Init image for img2img
    if (params.mode == IMG2IMG && input_image_buffer) {
        gen_params.init_image.width = params.width;
        gen_params.init_image.height = params.height;
        gen_params.init_image.channel = 3;
        gen_params.init_image.data = input_image_buffer;
    }

    // Generate
    sd_image_t* results = generate_image(sd_ctx, &gen_params);
    if (!results) {
        fprintf(stderr, "error: generation failed\n");
        free_sd_ctx(sd_ctx);
        free(input_image_buffer);
        free(control_image_buffer);
        return 1;
    }

    // Upscale if requested
    if (!params.esrgan_path.empty() && params.upscale_repeats > 0) {
        upscaler_ctx_t* upscaler_ctx = new_upscaler_ctx(params.esrgan_path.c_str(),
                                                         false, false,
                                                         params.n_threads, 128);
        if (upscaler_ctx) {
            for (int i = 0; i < params.batch_count; i++) {
                if (!results[i].data) continue;
                sd_image_t current = results[i];
                for (int u = 0; u < params.upscale_repeats; u++) {
                    sd_image_t upscaled = upscale(upscaler_ctx, current, 4);
                    if (!upscaled.data) break;
                    free(current.data);
                    current = upscaled;
                }
                results[i] = current;
            }
            free_upscaler_ctx(upscaler_ctx);
        }
    }

    // Save results
    size_t last = params.output_path.find_last_of(".");
    std::string base_name = last != std::string::npos ? params.output_path.substr(0, last) : params.output_path;
    for (int i = 0; i < params.batch_count; i++) {
        if (!results[i].data) continue;
        std::string path = i > 0 ? base_name + "_" + std::to_string(i + 1) + ".png" : base_name + ".png";
        stbi_write_png(path.c_str(), results[i].width, results[i].height, results[i].channel,
                       results[i].data, 0, get_image_params(params, params.seed + i).c_str());
        printf("saved: %s\n", path.c_str());
        free(results[i].data);
    }

    free(results);
    free_sd_ctx(sd_ctx);
    free(input_image_buffer);
    free(control_image_buffer);

    return 0;
}
