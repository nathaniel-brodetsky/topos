#pragma once

#include <map>
#include <vector>
#include <cstdint>

class OrderBookState {
public:
    void clear() {
        bids_.clear();
        asks_.clear();
    }

    void apply_bid_update(double price, double quantity) {
        if (quantity == 0.0) {
            bids_.erase(price);
        } else {
            bids_[price] = quantity;
        }
    }

    void apply_ask_update(double price, double quantity) {
        if (quantity == 0.0) {
            asks_.erase(price);
        } else {
            asks_[price] = quantity;
        }
    }

    struct Level {
        double price;
        double quantity;
    };

    std::vector<Level> top_bids(int n) const {
        std::vector<Level> result;
        result.reserve(n);
        int count = 0;
        for (auto it = bids_.rbegin(); it != bids_.rend() && count < n; ++it, ++count) {
            result.push_back({it->first, it->second});
        }
        return result;
    }

    std::vector<Level> top_asks(int n) const {
        std::vector<Level> result;
        result.reserve(n);
        int count = 0;
        for (auto it = asks_.begin(); it != asks_.end() && count < n; ++it, ++count) {
            result.push_back({it->first, it->second});
        }
        return result;
    }

    size_t n_bid_levels() const { return bids_.size(); }
    size_t n_ask_levels() const { return asks_.size(); }

private:
    std::map<double, double> bids_;
    std::map<double, double> asks_;
};
