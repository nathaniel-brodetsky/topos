#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

union Entry {
    int missing;
    float fvalue;
    int qvalue;
};

typedef void (*predict_fn_t)(Entry*, int, float*);

int main() {
    void* handle = dlopen("/tmp/xgb_model_compiled.so", RTLD_NOW);
    if (!handle) {
        printf("dlopen FAILED: %s\n", dlerror());
        return 1;
    }

    predict_fn_t predict_fn = (predict_fn_t)dlsym(handle, "predict");
    if (!predict_fn) {
        printf("dlsym FAILED: %s\n", dlerror());
        return 1;
    }

    printf("loaded predict() symbol successfully via dlopen/dlsym\n");

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const int n_features = 11;
    Entry data[n_features];
    for (int i = 0; i < n_features; ++i) {
        data[i].fvalue = dist(rng);
    }

    float result[1];

    const int n_warmup = 1000;
    const int n_trials = 1000000;

    for (int i = 0; i < n_warmup; ++i) {
        predict_fn(data, 0, result);
    }

    std::vector<double> latencies_us;
    latencies_us.reserve(n_trials);

    for (int i = 0; i < n_trials; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        predict_fn(data, 0, result);
        auto end = std::chrono::high_resolution_clock::now();
        latencies_us.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    std::sort(latencies_us.begin(), latencies_us.end());
    double mean = 0.0;
    for (double v : latencies_us) mean += v;
    mean /= latencies_us.size();

    double p50 = latencies_us[latencies_us.size() * 50 / 100];
    double p99 = latencies_us[latencies_us.size() * 99 / 100];
    double p999 = latencies_us[(size_t)(latencies_us.size() * 0.999)];

    printf("tl2cgen compiled tree, DIRECT C++ call (no Python at all):\n");
    printf("  mean=%.4f us  p50=%.4f us  p99=%.4f us  p999=%.4f us\n", mean, p50, p99, p999);
    printf("  result[0]=%.6f (sanity check)\n", result[0]);

    dlclose(handle);
    return 0;
}
