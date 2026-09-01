#pragma once

#include "order.hpp"
#include "spinlock.hpp"
#include <vector>
#include <cstdio>
#include <cmath>

constexpr double TAKER_FEE_RATE = 0.0004;
constexpr double MAKER_FEE_RATE = 0.0000;
constexpr double PRICE_EPS = 1e-9;

struct TradeEvent {
    double price;
    double quantity;
    bool sell_side_aggressor;
    uint64_t tick;
};

class ShadowMatcher {
public:
    ShadowMatcher() : next_order_id_(1) {}

    void lock() { lock_.lock(); }
    void unlock() { lock_.unlock(); }

    uint64_t place_limit_order(Side side, double price, double quantity,
                                double queue_ahead_volume, uint64_t current_tick) {
        SpinlockGuard guard(lock_);
        Order order;
        order.order_id = next_order_id_++;
        order.side = side;
        order.type = OrderType::LIMIT;
        order.price = price;
        order.original_quantity = quantity;
        order.remaining_quantity = quantity;
        order.queue_position = queue_ahead_volume;
        order.status = OrderStatus::OPEN;
        order.placed_tick = current_tick;
        order.filled_tick = 0;
        order.fill_price = 0.0;
        order.fee_paid = 0.0;

        open_orders_.push_back(order);

        printf("[matcher] LIMIT order placed. id=%lu side=%s price=%.2f qty=%.4f queue_position=%.4f\n",
               order.order_id, side == Side::BUY ? "BUY" : "SELL", price, quantity, queue_ahead_volume);

        return order.order_id;
    }

    void place_market_order(Side side, const double* ask_prices, const double* ask_volumes,
                             const double* bid_prices, const double* bid_volumes,
                             int n_levels, double quantity, uint64_t current_tick) {
        SpinlockGuard guard(lock_);
        Order order;
        order.order_id = next_order_id_++;
        order.side = side;
        order.type = OrderType::MARKET;
        order.original_quantity = quantity;
        order.status = OrderStatus::PENDING;
        order.placed_tick = current_tick;

        const double* prices = (side == Side::BUY) ? ask_prices : bid_prices;
        const double* volumes = (side == Side::BUY) ? ask_volumes : bid_volumes;

        double remaining = quantity;
        double total_cost = 0.0;
        double total_filled = 0.0;

        for (int i = 0; i < n_levels && remaining > PRICE_EPS; ++i) {
            double take = std::min(remaining, volumes[i]);
            total_cost += take * prices[i];
            total_filled += take;
            remaining -= take;
        }

        if (total_filled < PRICE_EPS) {
            order.status = OrderStatus::REJECTED;
            printf("[matcher] MARKET order REJECTED. id=%lu no liquidity available\n", order.order_id);
            filled_orders_.push_back(order);
            return;
        }

        double avg_fill_price = total_cost / total_filled;
        double fee = total_cost * TAKER_FEE_RATE;

        order.fill_price = avg_fill_price;
        order.remaining_quantity = quantity - total_filled;
        order.status = (order.remaining_quantity < PRICE_EPS) ? OrderStatus::FILLED : OrderStatus::PENDING;
        order.filled_tick = current_tick;
        order.fee_paid = fee;

        printf("[matcher] MARKET order FILLED. id=%lu side=%s requested_qty=%.4f filled_qty=%.4f avg_price=%.2f fee=%.4f slippage_levels_consumed=%s\n",
               order.order_id, side == Side::BUY ? "BUY" : "SELL", quantity, total_filled, avg_fill_price, fee,
               (remaining > PRICE_EPS) ? "YES (partial fill, book exhausted)" : "no");

        position_.apply_fill(side, avg_fill_price, total_filled, fee);
        filled_orders_.push_back(order);
    }

    void on_trade_batch(const std::vector<TradeEvent>& batch) {
        SpinlockGuard guard(lock_);
        for (const auto& ev : batch) {
            on_trade_locked(ev.price, ev.quantity, ev.sell_side_aggressor, ev.tick);
        }
    }

    void on_trade(double trade_price, double trade_qty, bool trade_was_sell_side_aggressor, uint64_t current_tick) {
        SpinlockGuard guard(lock_);
        on_trade_locked(trade_price, trade_qty, trade_was_sell_side_aggressor, current_tick);
    }

private:
    void on_trade_locked(double trade_price, double trade_qty, bool trade_was_sell_side_aggressor, uint64_t current_tick) {
        for (auto it = open_orders_.begin(); it != open_orders_.end();) {
            Order& order = *it;

            bool trade_hits_our_side =
                (order.side == Side::BUY && trade_was_sell_side_aggressor && std::abs(trade_price - order.price) < PRICE_EPS) ||
                (order.side == Side::SELL && !trade_was_sell_side_aggressor && std::abs(trade_price - order.price) < PRICE_EPS);

            if (!trade_hits_our_side) {
                ++it;
                continue;
            }

            double prev_queue = order.queue_position;
            order.queue_position -= trade_qty;

            printf("[matcher] Trade hit our price level. order_id=%lu trade_qty=%.4f queue_position: %.4f -> %.4f\n",
                   order.order_id, trade_qty, prev_queue, order.queue_position);

            if (order.queue_position <= PRICE_EPS) {
                double fill_qty = order.remaining_quantity;
                double fee = fill_qty * order.price * MAKER_FEE_RATE;

                order.status = OrderStatus::FILLED;
                order.filled_tick = current_tick;
                order.fill_price = order.price;
                order.fee_paid = fee;
                order.remaining_quantity = 0.0;

                printf("[matcher] Order FILLED (queue exhausted). id=%lu side=%s price=%.2f qty=%.4f fee=%.4f\n",
                       order.order_id, order.side == Side::BUY ? "BUY" : "SELL",
                       order.price, fill_qty, fee);

                position_.apply_fill(order.side, order.fill_price, fill_qty, fee);
                filled_orders_.push_back(order);
                it = open_orders_.erase(it);
            } else {
                ++it;
            }
        }
    }

public:
    bool cancel_order(uint64_t order_id, uint64_t current_tick) {
        SpinlockGuard guard(lock_);
        for (auto it = open_orders_.begin(); it != open_orders_.end(); ++it) {
            if (it->order_id == order_id) {
                it->status = OrderStatus::CANCELED;
                printf("[matcher] Order CANCELED. id=%lu\n", order_id);
                canceled_orders_.push_back(*it);
                open_orders_.erase(it);
                return true;
            }
        }
        return false;
    }

    void cancel_all(uint64_t current_tick) {
        SpinlockGuard guard(lock_);
        printf("[matcher] CANCEL ALL triggered. canceling %zu open orders\n", open_orders_.size());
        for (auto& order : open_orders_) {
            order.status = OrderStatus::CANCELED;
            canceled_orders_.push_back(order);
        }
        open_orders_.clear();
    }

    Position position() const {
        SpinlockGuard guard(const_cast<Spinlock&>(lock_));
        return position_;
    }
    uint64_t oldest_open_order_placed_tick() const {
        SpinlockGuard guard(const_cast<Spinlock&>(lock_));
        if (open_orders_.empty()) return 0;
        uint64_t oldest = open_orders_[0].placed_tick;
        for (auto& o : open_orders_) {
            if (o.placed_tick < oldest) oldest = o.placed_tick;
        }
        return oldest;
    }

    bool get_single_open_order_side(Side& out_side) const {
        SpinlockGuard guard(const_cast<Spinlock&>(lock_));
        if (open_orders_.empty()) return false;
        out_side = open_orders_[0].side;
        return true;
    }
    size_t n_open_orders() const {
        SpinlockGuard guard(const_cast<Spinlock&>(lock_));
        return open_orders_.size();
    }
    size_t n_filled_orders() const {
        SpinlockGuard guard(const_cast<Spinlock&>(lock_));
        return filled_orders_.size();
    }
    size_t n_canceled_orders() const {
        SpinlockGuard guard(const_cast<Spinlock&>(lock_));
        return canceled_orders_.size();
    }

    std::vector<Order> filled_orders() const {
        SpinlockGuard guard(const_cast<Spinlock&>(lock_));
        return filled_orders_;
    }

private:
    uint64_t next_order_id_;
    std::vector<Order> open_orders_;
    std::vector<Order> filled_orders_;
    std::vector<Order> canceled_orders_;
    Position position_;
    Spinlock lock_;
};
