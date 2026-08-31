#pragma once

#include <array>
#include <atomic>
#include <cstring>
#include "kmeans_calibration.hpp"

constexpr int RING_BUFFER_SIZE = 2000;

class QueryRingBuffer {
public:
    QueryRingBuffer() : write_idx_(0), filled_count_(0) {}

    void push(const std::array<float, CALIB_DIM>& query) {
        size_t idx = write_idx_.fetch_add(1, std::memory_order_relaxed) % RING_BUFFER_SIZE;
        buffer_[idx] = query;
        size_t count = filled_count_.load(std::memory_order_relaxed);
        if (count < RING_BUFFER_SIZE) {
            filled_count_.store(count + 1, std::memory_order_relaxed);
        }
    }

    std::vector<std::array<float, CALIB_DIM>> snapshot() const {
        size_t count = filled_count_.load(std::memory_order_relaxed);
        std::vector<std::array<float, CALIB_DIM>> result;
        result.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            result.push_back(buffer_[i]);
        }
        return result;
    }

    size_t size() const {
        return filled_count_.load(std::memory_order_relaxed);
    }

private:
    std::array<std::array<float, CALIB_DIM>, RING_BUFFER_SIZE> buffer_;
    std::atomic<size_t> write_idx_;
    std::atomic<size_t> filled_count_;
};
