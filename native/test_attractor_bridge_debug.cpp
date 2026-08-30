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
        for (int d = 0; d < CENTROID_DIM; ++d) initial_staging->centroids[k][d] = 0.0f;
        initial_staging->epsilon[k] = 1.0f;
        initial_staging->regime_id[k] = 0;
    }
    bridge.publish(1);

    std::atomic<bool> stop_flag{false};
    std::atomic<int> corruption_count{0};
    int first_corruption_report = 0;

    std::thread reader([&]() {
        int local_reports = 0;
        while (!stop_flag.load(std::memory_order_relaxed)) {
            const AttractorSnapshot* snap = bridge.current();
            uint64_t v = snap->version;
            int n = snap->n_centroids;

            bool mismatch_within_this_snapshot = false;
            float first_val = snap->centroids[0][0];
            for (int k = 0; k < n; ++k) {
                for (int d = 0; d < CENTROID_DIM; ++d) {
                    if (snap->centroids[k][d] != first_val) {
                        mismatch_within_this_snapshot = true;
                    }
                }
            }

            bool version_mismatch = (first_val != static_cast<float>(v));

            if (mismatch_within_this_snapshot) {
                corruption_count.fetch_add(1, std::memory_order_relaxed);
                if (local_reports < 3) {
                    printf("INTERNAL mismatch: version=%lu centroids[0][0]=%.1f but not all equal within same snapshot\n", v, first_val);
                    local_reports++;
                }
            } else if (version_mismatch) {
                corruption_count.fetch_add(1, std::memory_order_relaxed);
                if (local_reports < 3) {
                    printf("VERSION mismatch: version field=%lu but centroids[0][0]=%.1f (should match version)\n", v, first_val);
                    local_reports++;
                }
            }
        }
    });

    std::thread writer([&]() {
        for (uint64_t v = 2; v <= 100; ++v) {
            AttractorSnapshot* staging = bridge.staging();
            for (int k = 0; k < 3; ++k) {
                for (int d = 0; d < CENTROID_DIM; ++d) staging->centroids[k][d] = static_cast<float>(v);
                staging->epsilon[k] = 1.0f;
                staging->regime_id[k] = static_cast<int>(v % 3);
            }
            staging->n_centroids = 3;
            bridge.publish(v);
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    });

    writer.join();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    stop_flag.store(true, std::memory_order_relaxed);
    reader.join();

    printf("total corruption events: %d\n", corruption_count.load());
    return 0;
}
