#pragma once

#include <atomic>
#include <vector>
#include <array>
#include <cstring>
#include "kmeans_calibration.hpp"
#include "regime_labeling.hpp"

constexpr int MAX_LIVE_CENTROIDS = 128;

struct LiveAttractorSnapshot {
    uint64_t version;
    int n_centroids;
    std::array<float, MAX_LIVE_CENTROIDS * CALIB_DIM> centroids_flat;
    std::array<float, MAX_LIVE_CENTROIDS> epsilon;
    std::array<SemanticRegime, MAX_LIVE_CENTROIDS> regime;

    float centroid(int k, int d) const { return centroids_flat[k * CALIB_DIM + d]; }
};

class LiveAttractorBridge {
public:
    LiveAttractorBridge() {
        buffer_a_.version = 0;
        buffer_b_.version = 0;
        active_.store(&buffer_a_, std::memory_order_release);
        inactive_ = &buffer_b_;
    }

    const LiveAttractorSnapshot* current() const {
        return active_.load(std::memory_order_acquire);
    }

    void publish(const std::vector<std::array<float, CALIB_DIM>>& centroids,
                 const std::vector<float>& epsilon,
                 const std::vector<SemanticRegime>& regime,
                 uint64_t new_version) {
        LiveAttractorSnapshot* staging = inactive_;
        staging->n_centroids = static_cast<int>(centroids.size());
        for (size_t k = 0; k < centroids.size() && k < MAX_LIVE_CENTROIDS; ++k) {
            for (int d = 0; d < CALIB_DIM; ++d) {
                staging->centroids_flat[k * CALIB_DIM + d] = centroids[k][d];
            }
            staging->epsilon[k] = epsilon[k];
            staging->regime[k] = regime[k];
        }
        staging->version = new_version;

        LiveAttractorSnapshot* old_active = active_.load(std::memory_order_acquire);
        active_.store(staging, std::memory_order_release);
        inactive_ = old_active;
    }

private:
    LiveAttractorSnapshot buffer_a_;
    LiveAttractorSnapshot buffer_b_;
    std::atomic<LiveAttractorSnapshot*> active_;
    LiveAttractorSnapshot* inactive_;
};
