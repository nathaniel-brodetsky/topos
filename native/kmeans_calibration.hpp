#pragma once

#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdio>

constexpr int CALIB_DIM = 5;

struct CalibrationResult {
    std::vector<std::array<float, CALIB_DIM>> centroids;
    std::vector<float> epsilon;
    std::vector<int> final_assignment;
};

inline float squared_dist(const std::array<float, CALIB_DIM>& a, const std::array<float, CALIB_DIM>& b) {
    float sum = 0.0f;
    for (int d = 0; d < CALIB_DIM; ++d) {
        float diff = a[d] - b[d];
        sum += diff * diff;
    }
    return sum;
}

inline CalibrationResult run_kmeans_calibration(
    const std::vector<std::array<float, CALIB_DIM>>& data,
    int k,
    int n_iterations,
    float epsilon_percentile
) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> pick(0, data.size() - 1);

    std::vector<std::array<float, CALIB_DIM>> centroids(k);
    for (int i = 0; i < k; ++i) {
        centroids[i] = data[pick(rng)];
    }

    std::vector<int> assignment(data.size());

    for (int iter = 0; iter < n_iterations; ++iter) {
        for (size_t i = 0; i < data.size(); ++i) {
            int best_idx = 0;
            float best_dist = squared_dist(data[i], centroids[0]);
            for (int c = 1; c < k; ++c) {
                float d = squared_dist(data[i], centroids[c]);
                if (d < best_dist) { best_dist = d; best_idx = c; }
            }
            assignment[i] = best_idx;
        }

        std::vector<std::array<float, CALIB_DIM>> sums(k, {0, 0, 0, 0, 0});
        std::vector<int> counts(k, 0);

        for (size_t i = 0; i < data.size(); ++i) {
            int c = assignment[i];
            for (int d = 0; d < CALIB_DIM; ++d) sums[c][d] += data[i][d];
            counts[c]++;
        }

        for (int c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                for (int d = 0; d < CALIB_DIM; ++d) centroids[c][d] = sums[c][d] / counts[c];
            }
        }
    }

    std::vector<std::vector<float>> cluster_distances(k);
    for (size_t i = 0; i < data.size(); ++i) {
        int c = assignment[i];
        float dist = std::sqrt(squared_dist(data[i], centroids[c]));
        cluster_distances[c].push_back(dist);
    }

    std::vector<float> epsilon(k, 1.0f);
    for (int c = 0; c < k; ++c) {
        if (cluster_distances[c].empty()) continue;
        std::sort(cluster_distances[c].begin(), cluster_distances[c].end());
        size_t idx = static_cast<size_t>(cluster_distances[c].size() * epsilon_percentile);
        idx = std::min(idx, cluster_distances[c].size() - 1);
        epsilon[c] = cluster_distances[c][idx];
        if (epsilon[c] <= 0.0f) epsilon[c] = 1.0f;
    }

    return {centroids, epsilon, assignment};
}
