#pragma once

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <cstdint>

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;
using json = nlohmann::json;

struct DepthSnapshot {
    uint64_t last_update_id;
    json raw_bids;
    json raw_asks;
};

inline DepthSnapshot fetch_depth_snapshot(const std::string& symbol, int limit = 1000) {
    const std::string host = "fapi.binance.com";
    const std::string port = "443";
    const std::string target = "/fapi/v1/depth?symbol=" + symbol + "&limit=" + std::to_string(limit);

    net::io_context ioc;
    ssl::context ctx{ssl::context::tlsv12_client};
    ctx.set_default_verify_paths();

    tcp::resolver resolver{ioc};
    beast::ssl_stream<beast::tcp_stream> stream{ioc, ctx};

    if (!SSL_set_tlsext_host_name(stream.native_handle(), host.c_str())) {
        throw beast::system_error(
            beast::error_code(static_cast<int>(::ERR_get_error()), net::error::get_ssl_category()),
            "Failed to set SNI hostname");
    }

    auto const results = resolver.resolve(host, port);
    beast::get_lowest_layer(stream).connect(results);
    stream.handshake(ssl::stream_base::client);

    http::request<http::string_body> req{http::verb::get, target, 11};
    req.set(http::field::host, host);
    req.set(http::field::user_agent, "topos-rest-client/1.0");

    http::write(stream, req);

    beast::flat_buffer buffer;
    http::response<http::string_body> res;
    http::read(stream, buffer, res);

    if (res.result() != http::status::ok) {
        throw std::runtime_error("REST snapshot request failed with HTTP status " +
                                  std::to_string(res.result_int()) + ": " + res.body());
    }

    json j = json::parse(res.body());

    beast::error_code ec;
    stream.shutdown(ec);

    DepthSnapshot snapshot;
    snapshot.last_update_id = j["lastUpdateId"].get<uint64_t>();
    snapshot.raw_bids = j["bids"];
    snapshot.raw_asks = j["asks"];

    return snapshot;
}
