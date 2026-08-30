#pragma once

#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include "kmeans_calibration.hpp"

enum class SemanticRegime { EQUILIBRIUM, IMPULSE, TRANSITIONAL };

inline std::vector<SemanticRegime> label_centroids_by_pressure(
    const std::vector<std::array<float, CALIB_DIM>>& data,
    const std::vector<int>& assignment,
    int k,
    float impulse_percentile = 0.66f,
    float equilibrium_percentile = 0.33f
) {
    std::vector<double> pressure_sum(k, 0.0);
    std::vector<int> counts(k, 0);

    for (size_t i = 0; i < data.size(); ++i) {
        int c = assignment[i];
        float directed_pressure_proxy = data[i][1];
        pressure_sum[c] += std::fabs(directed_pressure_proxy);
        counts[c]++;
    }

    std::vector<float> mean_abs_pressure(k, 0.0f);
    for (int c = 0; c < k; ++c) {
        mean_abs_pressure[c] = (counts[c] > 0) ? static_cast<float>(pressure_sum[c] / counts[c]) : 0.0f;
    }

    std::vector<float> sorted_pressure = mean_abs_pressure;
    std::sort(sorted_pressure.begin(), sorted_pressure.end());

    size_t impulse_idx = static_cast<size_t>(sorted_pressure.size() * impulse_percentile);
    size_t equilibrium_idx = static_cast<size_t>(sorted_pressure.size() * equilibrium_percentile);
    impulse_idx = std::min(impulse_idx, sorted_pressure.size() - 1);
    equilibrium_idx = std::min(equilibrium_idx, sorted_pressure.size() - 1);

    float impulse_threshold = sorted_pressure[impulse_idx];
    float equilibrium_threshold = sorted_pressure[equilibrium_idx];

    std::vector<SemanticRegime> labels(k);
    for (int c = 0; c < k; ++c) {
        if (mean_abs_pressure[c] >= impulse_threshold) {
            labels[c] = SemanticRegime::IMPULSE;
        } else if (mean_abs_pressure[c] <= equilibrium_threshold) {
            labels[c] = SemanticRegime::EQUILIBRIUM;
        } else {
            labels[c] = SemanticRegime::TRANSITIONAL;
        }
    }

    return labels;
}
