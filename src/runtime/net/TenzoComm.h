#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>

namespace tenzo {
namespace net {

#pragma pack(push, 1)
struct PacketHeader {
    uint32_t magic;         // 0x544E5A4F ('TNZO')
    uint8_t  command;       // 1 = EXEC_LAYERS, 2 = RESULT_ACTIVATION, 3 = PING, 4 = PONG
    uint32_t seqId;         // Token / Request sequence ID
    uint16_t startLayer;    // Inclusive start layer
    uint16_t endLayer;      // Exclusive end layer
    uint32_t dataBytes;     // Payload size in bytes
};
#pragma pack(pop)

constexpr uint32_t TENZO_NET_MAGIC = 0x544E5A4F;

/// Minimalist, high-throughput binary client for sending activation buffers to remote nodes
class TenzoClient {
public:
    TenzoClient(const std::string& host, int port);
    ~TenzoClient();

    bool connect();
    bool connectToServer() { return connect(); }
    void disconnect();
    bool isConnected() const;

    /// Measure round-trip ping time in milliseconds
    double ping();

    /// Send input activation to remote node, execute [startLayer, endLayer), and receive output activation
    bool executeRemoteLayers(uint32_t seqId,
                             uint16_t startLayer,
                             uint16_t endLayer,
                             const float* inActivation,
                             size_t dim,
                             float* outActivation);

private:
    std::string host;
    int port;
    int socketFd;
};

/// Minimalist remote worker server that listens for activation packets and executes layers locally
class TenzoServer {
public:
    using LayerExecCallback = std::function<bool(uint16_t startLayer,
                                                 uint16_t endLayer,
                                                 const float* inAct,
                                                 size_t dim,
                                                 float* outAct)>;

    TenzoServer(int port, LayerExecCallback callback);
    ~TenzoServer();

    bool start();
    void stop();

private:
    int port;
    int serverFd;
    LayerExecCallback callback;
    std::atomic<bool> running;
    std::unique_ptr<std::thread> listenThread;

    void handleClient(int clientFd);
};

} // namespace net
} // namespace tenzo
