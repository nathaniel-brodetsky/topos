#include "attractor_bridge.hpp"
#include <thread>
#include <atomic>
#include <cstdio>
#include <chrono>

int main() {
    AttractorBridge bridge;

    AttractorSnapshot* initial_staging = bridge.staging();
    initial_staging->n_centroids = 3;
    for (int k = 0; k < 3; ++k) {
        for (int d = 0; d < CENTROID_DIM; ++d) initial_staging->centroids[k][d] = 1.0f;
        initial_staging->epsilon[k] = 1.0f;
        initial_staging->regime_id[k] = 0;
    }
    bridge.publish(1);

    std::atomic<bool> stop_flag{false};
    std::atomic<uint64_t> reader_iterations{0};
    std::atomic<uint64_t> max_version_seen{0};
    std::atomic<bool> corruption_detected{false};

    std::thread reader([&]() {
        while (!stop_flag.load(std::memory_order_relaxed)) {
            const AttractorSnapshot* snap = bridge.current();

            uint64_t v = snap->version;
            int n = snap->n_centroids;
            float expected = static_cast<float>(v);

            for (int k = 0; k < n; ++k) {
                for (int d = 0; d < CENTROID_DIM; ++d) {
                    if (snap->centroids[k][d] != expected) {
                        corruption_detected.store(true, std::memory_order_relaxed);
                    }
                }
            }

            if (v > max_version_seen.load(std::memory_order_relaxed)) {
                max_version_seen.store(v, std::memory_order_relaxed);
            }

            reader_iterations.fetch_add(1, std::memory_order_relaxed);
        }
    });

    std::thread writer([&]() {
        for (uint64_t v = 2; v <= 100; ++v) {
            AttractorSnapshot* staging = bridge.staging();
            staging->n_centroids = 3;
            for (int k = 0; k < 3; ++k) {
                for (int d = 0; d < CENTROID_DIM; ++d) staging->centroids[k][d] = static_cast<float>(v);
                staging->epsilon[k] = 1.0f;
                staging->regime_id[k] = static_cast<int>(v % 3);
            }
            bridge.publish(v);
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    });

    writer.join();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    stop_flag.store(true, std::memory_order_relaxed);
    reader.join();

    printf("reader iterations: %lu\n", reader_iterations.load());
    printf("max version seen by reader: %lu (expected 100)\n", max_version_seen.load());
    printf("corruption detected (torn read of centroid data): %s\n", corruption_detected.load() ? "YES - BUG" : "no");

    if (corruption_detected.load()) {
        printf("TEST FAILED\n");
        return 1;
    }
    printf("TEST PASSED: no torn reads observed across %lu iterations while writer published 99 updates\n", reader_iterations.load());

    return 0;
}
