#include <nlohmann/json.hpp>
#include "orderbook_state.hpp"
#include <iostream>
#include <string>

using json = nlohmann::json;

void apply_depth_update(OrderBookState& book, const std::string& message) {
    json j = json::parse(message);

    for (auto& level : j["b"]) {
        double price = std::stod(level[0].get<std::string>());
        double quantity = std::stod(level[1].get<std::string>());
        book.apply_bid_update(price, quantity);
    }

    for (auto& level : j["a"]) {
        double price = std::stod(level[0].get<std::string>());
        double quantity = std::stod(level[1].get<std::string>());
        book.apply_ask_update(price, quantity);
    }
}

int main() {
    OrderBookState book;

    std::string msg1 = R"({"e":"depthUpdate","b":[["78747.90","7.986"],["78745.30","0.026"],["78744.90","0.032"]],"a":[["78749.00","0.027"],["78750.50","0.002"],["78751.00","0.000"]]})";
    std::string msg2 = R"({"e":"depthUpdate","b":[["78747.90","7.966"],["78745.30","0.000"]],"a":[["78750.50","0.005"],["78749.00","0.000"]]})";

    apply_depth_update(book, msg1);
    std::cout << "After msg1: bid_levels=" << book.n_bid_levels() << " ask_levels=" << book.n_ask_levels() << "\n";

    auto top_bids = book.top_bids(3);
    for (auto& lvl : top_bids) std::cout << "  bid: " << lvl.price << " @ " << lvl.quantity << "\n";
    auto top_asks = book.top_asks(3);
    for (auto& lvl : top_asks) std::cout << "  ask: " << lvl.price << " @ " << lvl.quantity << "\n";

    apply_depth_update(book, msg2);
    std::cout << "\nAfter msg2 (78745.30 bid removed, 78749.00 ask removed, 78750.50 ask qty updated):\n";
    std::cout << "bid_levels=" << book.n_bid_levels() << " ask_levels=" << book.n_ask_levels() << "\n";

    top_bids = book.top_bids(3);
    for (auto& lvl : top_bids) std::cout << "  bid: " << lvl.price << " @ " << lvl.quantity << "\n";
    top_asks = book.top_asks(3);
    for (auto& lvl : top_asks) std::cout << "  ask: " << lvl.price << " @ " << lvl.quantity << "\n";

    return 0;
}
