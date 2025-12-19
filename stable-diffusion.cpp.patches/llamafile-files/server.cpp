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

// sdfile HTTP server implementation - OpenAI compatible API

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <exception>

#include "thirdparty/httplib.h"
#include "llama.cpp/vendor/nlohmann/json.hpp"
#include "stable-diffusion.h"

#include "third_party/stb/stb_image.h"
#include "third_party/stb/stb_image_write.h"

#include "llamafile/llamafile.h"
// From llama.cpp/common/common.h - declared here to avoid heavy includes
int32_t cpu_get_num_math();

using json = nlohmann::json;

// ----------------------- helpers -----------------------
static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

static std::string base64_encode(const std::vector<uint8_t>& bytes) {
    std::string ret;
    int val = 0, valb = -6;
    for (uint8_t c : bytes) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            ret.push_back(base64_chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6)
        ret.push_back(base64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
    while (ret.size() % 4)
        ret.push_back('=');
    return ret;
}

static std::string iso_timestamp_now() {
    using namespace std::chrono;
    auto now      = system_clock::now();
    std::time_t t = system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

enum class ImageFormat { JPEG, PNG };

static std::vector<uint8_t> write_image_to_vector(
    ImageFormat format,
    const uint8_t* image,
    int width,
    int height,
    int channels,
    int quality = 90) {
    std::vector<uint8_t> buffer;

    auto write_func = [](void* context, void* data, int size) {
        std::vector<uint8_t>* buf = reinterpret_cast<std::vector<uint8_t>*>(context);
        uint8_t* src = reinterpret_cast<uint8_t*>(data);
        buf->insert(buf->end(), src, src + size);
    };

    int result = 0;
    switch (format) {
        case ImageFormat::JPEG:
            result = stbi_write_jpg_to_func(write_func, &buffer, width, height, channels, image, quality);
            break;
        case ImageFormat::PNG:
            result = stbi_write_png_to_func(write_func, &buffer, width, height, channels, image, width * channels);
            break;
    }

    if (!result) {
        buffer.clear();
    }

    return buffer;
}

struct SDServerParams {
    std::string listen_ip = "127.0.0.1";
    int listen_port = 7860;

    // Model paths
    std::string model_path;
    std::string diffusion_model_path;
    std::string vae_path;
    std::string taesd_path;
    std::string controlnet_path;
    std::string embeddings_path;

    // Text encoder paths (for SD3/Flux/etc)
    std::string clip_l_path;
    std::string clip_g_path;
    std::string t5xxl_path;
    std::string llm_path;
    std::string llm_vision_path;

    int n_threads = -1;
    sd_type_t wtype = SD_TYPE_COUNT;
    rng_type_t rng_type = CUDA_RNG;
    bool verbose = false;
    bool color = false;
    bool clip_on_cpu = false;
    bool vae_on_cpu = false;
    bool control_net_cpu = false;
    bool offload_to_cpu = false;
    bool diffusion_fa = false;
    bool vae_tiling = false;

    // Default generation params
    int default_width = 512;
    int default_height = 512;
    int default_steps = 20;
    float default_cfg = 7.0f;
    float default_guidance = 3.5f;  // distilled guidance for Flux/etc
    sample_method_t default_sample_method = EULER_A_SAMPLE_METHOD;
    scheduler_t default_scheduler = DISCRETE_SCHEDULER;
};

static void parse_server_args(int argc, const char** argv, SDServerParams& params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--host") {
            if (++i < argc) params.listen_ip = argv[i];
        } else if (arg == "--port") {
            if (++i < argc) params.listen_port = std::stoi(argv[i]);
        } else if (arg == "-m" || arg == "--model") {
            if (++i < argc) params.model_path = argv[i];
        } else if (arg == "--diffusion-model") {
            if (++i < argc) params.diffusion_model_path = argv[i];
        } else if (arg == "--vae") {
            if (++i < argc) params.vae_path = argv[i];
        } else if (arg == "--taesd" || arg == "--tae") {
            if (++i < argc) params.taesd_path = argv[i];
        } else if (arg == "--clip_l") {
            if (++i < argc) params.clip_l_path = argv[i];
        } else if (arg == "--clip_g") {
            if (++i < argc) params.clip_g_path = argv[i];
        } else if (arg == "--t5xxl") {
            if (++i < argc) params.t5xxl_path = argv[i];
        } else if (arg == "--llm" || arg == "--qwen2vl") {
            if (++i < argc) params.llm_path = argv[i];
        } else if (arg == "--llm_vision" || arg == "--qwen2vl_vision") {
            if (++i < argc) params.llm_vision_path = argv[i];
        } else if (arg == "--control-net") {
            if (++i < argc) params.controlnet_path = argv[i];
        } else if (arg == "--embd-dir") {
            if (++i < argc) params.embeddings_path = argv[i];
        } else if (arg == "-t" || arg == "--threads") {
            if (++i < argc) params.n_threads = std::stoi(argv[i]);
        } else if (arg == "--type") {
            if (++i < argc) params.wtype = str_to_sd_type(argv[i]);
        } else if (arg == "--rng") {
            if (++i < argc) params.rng_type = str_to_rng_type(argv[i]);
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else if (arg == "--color") {
            params.color = true;
        } else if (arg == "--clip-on-cpu") {
            params.clip_on_cpu = true;
        } else if (arg == "--vae-on-cpu") {
            params.vae_on_cpu = true;
        } else if (arg == "--control-net-cpu") {
            params.control_net_cpu = true;
        } else if (arg == "--offload-to-cpu") {
            params.offload_to_cpu = true;
        } else if (arg == "--diffusion-fa") {
            params.diffusion_fa = true;
        } else if (arg == "--vae-tiling") {
            params.vae_tiling = true;
        } else if (arg == "-W" || arg == "--width") {
            if (++i < argc) params.default_width = std::stoi(argv[i]);
        } else if (arg == "-H" || arg == "--height") {
            if (++i < argc) params.default_height = std::stoi(argv[i]);
        } else if (arg == "--steps") {
            if (++i < argc) params.default_steps = std::stoi(argv[i]);
        } else if (arg == "--cfg-scale") {
            if (++i < argc) params.default_cfg = std::stof(argv[i]);
        } else if (arg == "--guidance") {
            if (++i < argc) params.default_guidance = std::stof(argv[i]);
        }
    }

    if (params.n_threads <= 0) {
        params.n_threads = cpu_get_num_math();
    }
}

static bool server_verbose = false;

static void server_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    if (!log || (!server_verbose && level <= SD_LOG_DEBUG)) {
        return;
    }

    int tag_color;
    const char* level_str;
    FILE* out_stream = (level == SD_LOG_ERROR) ? stderr : stdout;

    switch (level) {
        case SD_LOG_DEBUG: tag_color = 37; level_str = "DEBUG"; break;
        case SD_LOG_INFO:  tag_color = 34; level_str = "INFO";  break;
        case SD_LOG_WARN:  tag_color = 35; level_str = "WARN";  break;
        case SD_LOG_ERROR: tag_color = 31; level_str = "ERROR"; break;
        default:           tag_color = 33; level_str = "?????"; break;
    }

    fprintf(out_stream, "\033[%d;1m[%-5s]\033[0m %s", tag_color, level_str, log);
    fflush(out_stream);
}

int sd_server_main(int argc, const char** argv) {
    SDServerParams params;
    parse_server_args(argc, argv, params);

    server_verbose = params.verbose;
    sd_set_log_callback(server_log_cb, nullptr);

    printf("[INFO] sdfile server starting...\n");
    if (!params.model_path.empty())
        printf("[INFO] Model: %s\n", params.model_path.c_str());
    if (!params.diffusion_model_path.empty())
        printf("[INFO] Diffusion Model: %s\n", params.diffusion_model_path.c_str());
    if (!params.llm_path.empty())
        printf("[INFO] LLM: %s\n", params.llm_path.c_str());
    printf("[INFO] Threads: %d\n", params.n_threads);

    // Create SD context
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
    ctx_params.vae_decode_only = true;
    ctx_params.n_threads = params.n_threads;
    ctx_params.wtype = params.wtype;
    ctx_params.rng_type = params.rng_type;
    ctx_params.offload_params_to_cpu = params.offload_to_cpu;
    ctx_params.keep_clip_on_cpu = params.clip_on_cpu;
    ctx_params.keep_vae_on_cpu = params.vae_on_cpu;
    ctx_params.keep_control_net_on_cpu = params.control_net_cpu;
    ctx_params.diffusion_flash_attn = params.diffusion_fa;

    sd_ctx_t* sd_ctx = new_sd_ctx(&ctx_params);
    if (!sd_ctx) {
        fprintf(stderr, "[ERROR] Failed to create SD context\n");
        return 1;
    }

    printf("[INFO] Model loaded successfully\n");

    std::mutex sd_ctx_mutex;
    httplib::Server svr;

    // CORS handling
    svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        std::string origin = req.get_header_value("Origin");
        if (origin.empty()) origin = "*";
        res.set_header("Access-Control-Allow-Origin", origin);
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods", "*");
        res.set_header("Access-Control-Allow-Headers", "*");

        if (req.method == "OPTIONS") {
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    // Health check
    svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"ok":true,"service":"sdfile-http"})", "application/json");
    });

    // Models endpoint
    svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        json r;
        r["data"] = json::array();
        r["data"].push_back({{"id", "sdfile-local"}, {"object", "model"}, {"owned_by", "local"}});
        res.set_content(r.dump(), "application/json");
    });

    // OpenAI-compatible image generation endpoint
    svr.Post("/v1/images/generations", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            if (req.body.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"empty body"})", "application/json");
                return;
            }

            json j = json::parse(req.body);
            std::string prompt = j.value("prompt", "");
            int n = std::max(1, std::min(8, j.value("n", 1)));
            std::string size = j.value("size", "");
            std::string output_format = j.value("output_format", "png");
            int output_quality = j.value("output_compression", 100);

            int width = params.default_width;
            int height = params.default_height;
            if (!size.empty()) {
                auto pos = size.find('x');
                if (pos != std::string::npos) {
                    try {
                        width = std::stoi(size.substr(0, pos));
                        height = std::stoi(size.substr(pos + 1));
                    } catch (...) {}
                }
            }

            // Ensure dimensions are multiples of 64
            width = (width / 64) * 64;
            height = (height / 64) * 64;
            if (width <= 0) width = 512;
            if (height <= 0) height = 512;

            if (prompt.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"prompt required"})", "application/json");
                return;
            }

            if (output_format != "png" && output_format != "jpeg") {
                output_format = "png";
            }

            // Set up generation parameters
            sd_img_gen_params_t gen_params;
            sd_img_gen_params_init(&gen_params);
            gen_params.prompt = prompt.c_str();
            gen_params.negative_prompt = j.value("negative_prompt", "").c_str();
            gen_params.width = width;
            gen_params.height = height;
            gen_params.batch_count = n;
            gen_params.seed = j.value("seed", (int64_t)-1);

            if (gen_params.seed < 0) {
                srand((int)time(NULL));
                gen_params.seed = rand();
            }

            sd_sample_params_init(&gen_params.sample_params);
            gen_params.sample_params.sample_method = params.default_sample_method;
            gen_params.sample_params.scheduler = params.default_scheduler;
            gen_params.sample_params.sample_steps = j.value("steps", params.default_steps);
            gen_params.sample_params.guidance.txt_cfg = j.value("cfg_scale", params.default_cfg);
            gen_params.sample_params.guidance.distilled_guidance = j.value("guidance", params.default_guidance);

            // VAE tiling
            gen_params.vae_tiling_params.enabled = params.vae_tiling;

            // Generate
            sd_image_t* results = nullptr;
            {
                std::lock_guard<std::mutex> lock(sd_ctx_mutex);
                results = generate_image(sd_ctx, &gen_params);
            }

            json out;
            out["created"] = iso_timestamp_now();
            out["data"] = json::array();
            out["output_format"] = output_format;

            if (results) {
                for (int i = 0; i < n; i++) {
                    if (!results[i].data) continue;

                    auto image_bytes = write_image_to_vector(
                        output_format == "jpeg" ? ImageFormat::JPEG : ImageFormat::PNG,
                        results[i].data, results[i].width, results[i].height,
                        results[i].channel, output_quality);

                    if (!image_bytes.empty()) {
                        std::string b64 = base64_encode(image_bytes);
                        json item;
                        item["b64_json"] = b64;
                        out["data"].push_back(item);
                    }

                    free(results[i].data);
                }
                free(results);
            }

            res.set_content(out.dump(), "application/json");
            res.status = 200;

        } catch (const std::exception& e) {
            res.status = 500;
            json err;
            err["error"] = "server_error";
            err["message"] = e.what();
            res.set_content(err.dump(), "application/json");
        }
    });

    printf("[INFO] Listening on %s:%d\n", params.listen_ip.c_str(), params.listen_port);
    printf("[INFO] API endpoint: POST /v1/images/generations\n");

    svr.listen(params.listen_ip, params.listen_port);

    free_sd_ctx(sd_ctx);
    return 0;
}
