#include "shadow_matcher.hpp"
#include <cassert>
#include <iostream>

int main() {
    ShadowMatcher matcher;

    std::cout << "=== TEST 1: Limit order queue tracking ===\n";
    uint64_t order_id = matcher.place_limit_order(Side::BUY, 100.0, 5.0, 10.0, 1);
    assert(matcher.n_open_orders() == 1);

    matcher.on_trade(100.0, 3.0, true, 2);
    assert(matcher.n_open_orders() == 1);

    matcher.on_trade(100.0, 4.0, true, 3);
    assert(matcher.n_open_orders() == 1);
    std::cout << "queue position after 7.0 total traded (should still be open, 3.0 remaining ahead of us): OK\n";

    matcher.on_trade(100.0, 3.5, true, 4);
    assert(matcher.n_open_orders() == 0);
    assert(matcher.n_filled_orders() == 1);
    std::cout << "PASS: order filled after queue exhausted\n\n";

    std::cout << "=== TEST 2: Market order with slippage across levels ===\n";
    ShadowMatcher matcher2;
    double ask_prices[3] = {101.0, 101.5, 102.0};
    double ask_volumes[3] = {2.0, 2.0, 2.0};
    double bid_prices[3] = {100.0, 99.5, 99.0};
    double bid_volumes[3] = {2.0, 2.0, 2.0};

    matcher2.place_market_order(Side::BUY, ask_prices, ask_volumes, bid_prices, bid_volumes, 3, 5.0, 1);
    const Position& pos = matcher2.position();
    std::cout << "position net_quantity=" << pos.net_quantity << " avg_entry_price=" << pos.avg_entry_price << "\n";
    assert(std::abs(pos.net_quantity - 5.0) < 1e-6);
    std::cout << "PASS: market order consumed multiple levels\n\n";

    std::cout << "=== TEST 3: Cancel all ===\n";
    ShadowMatcher matcher3;
    matcher3.place_limit_order(Side::BUY, 100.0, 1.0, 5.0, 1);
    matcher3.place_limit_order(Side::SELL, 101.0, 1.0, 5.0, 1);
    assert(matcher3.n_open_orders() == 2);
    matcher3.cancel_all(2);
    assert(matcher3.n_open_orders() == 0);
    assert(matcher3.n_canceled_orders() == 2);
    std::cout << "PASS: cancel_all clears all open orders\n\n";

    std::cout << "ALL TESTS PASSED\n";
    return 0;
}
