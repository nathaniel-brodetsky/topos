#pragma once

#include <atomic>
#include <cstring>
#include <cstdio>

constexpr int MAX_CENTROIDS = 128;
constexpr int CENTROID_DIM = 5;

struct AttractorSnapshot {
    int n_centroids;
    float centroids[MAX_CENTROIDS][CENTROID_DIM];
    float epsilon[MAX_CENTROIDS];
    int regime_id[MAX_CENTROIDS];
    uint64_t version;
};

class AttractorBridge {
public:
    AttractorBridge() {
        buffer_a_.version = 0;
        buffer_b_.version = 0;
        active_.store(&buffer_a_, std::memory_order_release);
        inactive_ = &buffer_b_;
    }

    const AttractorSnapshot* current() const {
        return active_.load(std::memory_order_acquire);
    }

    AttractorSnapshot* staging() {
        return inactive_;
    }

    void publish(uint64_t new_version) {
        inactive_->version = new_version;
        AttractorSnapshot* old_active = active_.load(std::memory_order_acquire);
        active_.store(inactive_, std::memory_order_release);
        inactive_ = old_active;
    }

private:
    AttractorSnapshot buffer_a_;
    AttractorSnapshot buffer_b_;
    std::atomic<AttractorSnapshot*> active_;
    AttractorSnapshot* inactive_;
};