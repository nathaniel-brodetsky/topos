#pragma once

#include <cstdint>
#include <atomic>
#include <cmath>
#include <algorithm>

enum class Side { BUY, SELL };
enum class OrderType { MARKET, LIMIT };
enum class OrderStatus { PENDING, OPEN, FILLED, CANCELED, REJECTED };

struct Order {
    uint64_t order_id;
    Side side;
    OrderType type;
    double price;
    double original_quantity;
    double remaining_quantity;
    double queue_position;
    OrderStatus status;
    uint64_t placed_tick;
    uint64_t filled_tick;
    double fill_price;
    double fee_paid;

    bool is_maker() const { return type == OrderType::LIMIT; }
};

struct Position {
    double net_quantity = 0.0;
    double avg_entry_price = 0.0;
    double realized_pnl = 0.0;
    double fees_paid_total = 0.0;

    void apply_fill(Side side, double fill_price, double fill_qty, double fee) {
        double signed_qty = (side == Side::BUY) ? fill_qty : -fill_qty;

        if (net_quantity == 0.0 || (net_quantity > 0) == (signed_qty > 0)) {
            double total_cost = avg_entry_price * std::abs(net_quantity) + fill_price * fill_qty;
            net_quantity += signed_qty;
            if (net_quantity != 0.0) {
                avg_entry_price = total_cost / std::abs(net_quantity);
            }
        } else {
            double closing_qty = std::min(std::abs(signed_qty), std::abs(net_quantity));
            double pnl_per_unit = (net_quantity > 0) ? (fill_price - avg_entry_price) : (avg_entry_price - fill_price);
            realized_pnl += pnl_per_unit * closing_qty;

            net_quantity += signed_qty;

            if ((net_quantity > 0) != (net_quantity - signed_qty > 0) && net_quantity != 0.0) {
                avg_entry_price = fill_price;
            }
        }

        fees_paid_total += fee;
    }

    double net_pnl() const {
        return realized_pnl - fees_paid_total;
    }
};
