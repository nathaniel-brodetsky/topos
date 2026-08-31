#include <nlohmann/json.hpp>
#include "orderbook_state.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

using json = nlohmann::json;

struct BufferedEvent {
    uint64_t U, u, pu;
    json raw;
};

struct MockSnapshot {
    uint64_t last_update_id;
    json raw_bids;
    json raw_asks;
};

json make_level_array(std::vector<std::pair<double,double>> levels) {
    json arr = json::array();
    for (auto& [p, q] : levels) {
        std::ostringstream ps, qs;
        ps << p; qs << q;
        arr.push_back({ps.str(), qs.str()});
    }
    return arr;
}

json make_diff_event(uint64_t U, uint64_t u, uint64_t pu,
                      std::vector<std::pair<double,double>> bids,
                      std::vector<std::pair<double,double>> asks) {
    json j;
    j["U"] = U;
    j["u"] = u;
    j["pu"] = pu;
    j["b"] = make_level_array(bids);
    j["a"] = make_level_array(asks);
    return j;
}

void apply_full_snapshot(OrderBookState& book, const MockSnapshot& snap) {
    book.clear();
    for (auto& level : snap.raw_bids) {
        double price = std::stod(level[0].get<std::string>());
        double qty = std::stod(level[1].get<std::string>());
        if (qty > 0.0) book.apply_bid_update(price, qty);
    }
    for (auto& level : snap.raw_asks) {
        double price = std::stod(level[0].get<std::string>());
        double qty = std::stod(level[1].get<std::string>());
        if (qty > 0.0) book.apply_ask_update(price, qty);
    }
}

void apply_diff_event(OrderBookState& book, const json& j) {
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

enum class ReconciliationResult { SUCCESS, GAP_DETECTED_REBOOT_REQUIRED };

ReconciliationResult reconcile(
    OrderBookState& book,
    const std::vector<BufferedEvent>& event_buffer,
    const MockSnapshot& snapshot,
    int& discarded,
    int& applied
) {
    apply_full_snapshot(book, snapshot);
    discarded = 0;
    applied = 0;
    bool first_applied = false;
    uint64_t last_u = 0;

    for (auto& ev : event_buffer) {
        if (ev.u <= snapshot.last_update_id) {
            discarded++;
            continue;
        }

        if (!first_applied) {
            bool brackets_correctly =
                (ev.U <= snapshot.last_update_id + 1) &&
                (snapshot.last_update_id + 1 <= ev.u);
            if (!brackets_correctly) {
                std::cerr << "  [WARNING] first event does not bracket lastUpdateId+1: U="
                          << ev.U << " u=" << ev.u << " lastUpdateId+1=" << (snapshot.last_update_id + 1) << "\n";
            }
            apply_diff_event(book, ev.raw);
            first_applied = true;
            last_u = ev.u;
            applied++;
            continue;
        }

        if (ev.pu != last_u) {
            std::cerr << "  [GAP DETECTED] event pu=" << ev.pu << " != last applied u=" << last_u
                      << " -- ABORTING reconciliation, reboot required\n";
            return ReconciliationResult::GAP_DETECTED_REBOOT_REQUIRED;
        }

        apply_diff_event(book, ev.raw);
        last_u = ev.u;
        applied++;
    }

    return ReconciliationResult::SUCCESS;
}

bool test_scenario_1_stale_events_discarded() {
    std::cout << "\n=== SCENARIO 1: events before snapshot must be discarded ===\n";

    MockSnapshot snapshot;
    snapshot.last_update_id = 1000;
    snapshot.raw_bids = make_level_array({{100.0, 5.0}, {99.5, 3.0}});
    snapshot.raw_asks = make_level_array({{100.5, 4.0}, {101.0, 2.0}});

    std::vector<BufferedEvent> buffer;
    buffer.push_back({900, 950, 899, make_diff_event(900, 950, 899, {{100.0, 99.0}}, {})});
    buffer.push_back({951, 999, 950, make_diff_event(951, 999, 950, {{100.0, 88.0}}, {})});
    buffer.push_back({1000, 1005, 999, make_diff_event(1000, 1005, 999, {{99.0, 1.0}}, {})});
    buffer.push_back({1006, 1010, 1005, make_diff_event(1006, 1010, 1005, {}, {{102.0, 1.5}})});

    OrderBookState book;
    int discarded, applied;
    auto result = reconcile(book, buffer, snapshot, discarded, applied);

    bool ok = (result == ReconciliationResult::SUCCESS) && (discarded == 2) && (applied == 2);
    std::cout << "  discarded=" << discarded << " (expected 2), applied=" << applied << " (expected 2)\n";
    std::cout << "  result: " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool test_scenario_2_clean_bracket() {
    std::cout << "\n=== SCENARIO 2: clean bracket, no gaps, full reconciliation ===\n";

    MockSnapshot snapshot;
    snapshot.last_update_id = 2000;
    snapshot.raw_bids = make_level_array({{100.0, 5.0}});
    snapshot.raw_asks = make_level_array({{100.5, 4.0}});

    std::vector<BufferedEvent> buffer;
    buffer.push_back({1998, 2003, 1997, make_diff_event(1998, 2003, 1997, {{100.0, 6.0}}, {})});
    buffer.push_back({2004, 2010, 2003, make_diff_event(2004, 2010, 2003, {{99.5, 2.0}}, {})});
    buffer.push_back({2011, 2015, 2010, make_diff_event(2011, 2015, 2010, {}, {{101.0, 1.0}})});

    OrderBookState book;
    int discarded, applied;
    auto result = reconcile(book, buffer, snapshot, discarded, applied);

    bool ok = (result == ReconciliationResult::SUCCESS) && (discarded == 0) && (applied == 3)
              && (book.n_bid_levels() == 2) && (book.n_ask_levels() == 2);

    std::cout << "  discarded=" << discarded << " (expected 0), applied=" << applied << " (expected 3)\n";
    std::cout << "  final bid_levels=" << book.n_bid_levels() << " (expected 2), ask_levels="
              << book.n_ask_levels() << " (expected 2)\n";
    std::cout << "  result: " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool test_scenario_3_gap_triggers_reboot() {
    std::cout << "\n=== SCENARIO 3: intentional gap must trigger reboot, not silent corruption ===\n";

    MockSnapshot snapshot;
    snapshot.last_update_id = 3000;
    snapshot.raw_bids = make_level_array({{100.0, 5.0}});
    snapshot.raw_asks = make_level_array({{100.5, 4.0}});

    std::vector<BufferedEvent> buffer;
    buffer.push_back({2998, 3005, 2997, make_diff_event(2998, 3005, 2997, {{100.0, 6.0}}, {})});
    buffer.push_back({3050, 3060, 3049, make_diff_event(3050, 3060, 3049, {{99.0, 1.0}}, {})});
    buffer.push_back({3061, 3070, 3060, make_diff_event(3061, 3070, 3060, {}, {{102.0, 1.0}})});

    OrderBookState book;
    int discarded, applied;
    auto result = reconcile(book, buffer, snapshot, discarded, applied);

    bool ok = (result == ReconciliationResult::GAP_DETECTED_REBOOT_REQUIRED) && (applied == 1);

    std::cout << "  applied before gap=" << applied << " (expected 1)\n";
    std::cout << "  result code: " << (result == ReconciliationResult::GAP_DETECTED_REBOOT_REQUIRED ? "GAP_DETECTED_REBOOT_REQUIRED" : "SUCCESS")
              << " (expected GAP_DETECTED_REBOOT_REQUIRED)\n";
    std::cout << "  result: " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool test_scenario_4_reboot_recovers_with_fresh_snapshot() {
    std::cout << "\n=== SCENARIO 4: after gap, a fresh snapshot + new buffer recovers cleanly ===\n";

    MockSnapshot fresh_snapshot;
    fresh_snapshot.last_update_id = 3070;
    fresh_snapshot.raw_bids = make_level_array({{99.0, 1.0}, {100.0, 6.0}});
    fresh_snapshot.raw_asks = make_level_array({{102.0, 1.0}});

    std::vector<BufferedEvent> new_buffer;
    new_buffer.push_back({3068, 3075, 3067, make_diff_event(3068, 3075, 3067, {{100.0, 7.0}}, {})});

    OrderBookState book;
    int discarded, applied;
    auto result = reconcile(book, new_buffer, fresh_snapshot, discarded, applied);

    bool ok = (result == ReconciliationResult::SUCCESS) && (applied == 1);
    std::cout << "  result: " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

int main() {
    std::cout << "TOPOS Reconciliation Logic Stress Test (synthetic, no live network dependency)\n";

    bool s1 = test_scenario_1_stale_events_discarded();
    bool s2 = test_scenario_2_clean_bracket();
    bool s3 = test_scenario_3_gap_triggers_reboot();
    bool s4 = test_scenario_4_reboot_recovers_with_fresh_snapshot();

    std::cout << "\n=== SUMMARY ===\n";
    std::cout << "Scenario 1 (stale discard):      " << (s1 ? "PASS" : "FAIL") << "\n";
    std::cout << "Scenario 2 (clean bracket):      " << (s2 ? "PASS" : "FAIL") << "\n";
    std::cout << "Scenario 3 (gap detection):      " << (s3 ? "PASS" : "FAIL") << "\n";
    std::cout << "Scenario 4 (reboot recovery):    " << (s4 ? "PASS" : "FAIL") << "\n";

    bool all_pass = s1 && s2 && s3 && s4;
    std::cout << "\n" << (all_pass ? "ALL SCENARIOS PASSED" : "SOME SCENARIOS FAILED") << "\n";

    return all_pass ? 0 : 1;
}
