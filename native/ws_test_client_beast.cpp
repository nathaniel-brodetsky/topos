#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <cstdlib>
#include <iostream>
#include <string>

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;

int main() {
    const std::string host = "fstream.binance.com";
    const std::string port = "443";
    const std::string target = "/ws/btcusdt@depth@100ms";

    try {
        net::io_context ioc;
        ssl::context ctx{ssl::context::tlsv12_client};
        ctx.set_default_verify_paths();

        tcp::resolver resolver{ioc};
        websocket::stream<beast::ssl_stream<beast::tcp_stream>> ws{ioc, ctx};

        auto const results = resolver.resolve(host, port);

        auto ep = beast::get_lowest_layer(ws).connect(results);
        std::cout << "TCP connected to " << host << ":" << ep.port() << "\n";

        if (!SSL_set_tlsext_host_name(ws.next_layer().native_handle(), host.c_str())) {
            throw beast::system_error(
                beast::error_code(static_cast<int>(::ERR_get_error()), net::error::get_ssl_category()),
                "Failed to set SNI hostname");
        }

        ws.next_layer().handshake(ssl::stream_base::client);
        std::cout << "TLS handshake complete\n";

        ws.set_option(websocket::stream_base::decorator(
            [](websocket::request_type& req) {
                req.set(http::field::user_agent, "topos-ws-client/1.0");
            }));

        ws.handshake(host, target);
        std::cout << "WebSocket handshake complete, subscribed to " << target << "\n";

        for (int i = 0; i < 5; ++i) {
            beast::flat_buffer buffer;
            ws.read(buffer);
            std::cout << "[" << i << "] " << beast::make_printable(buffer.data()) << "\n\n";
        }

        ws.close(websocket::close_code::normal);
        std::cout << "Closed cleanly\n";

    } catch (std::exception const& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
