#include "TenzoComm.h"
#include <iostream>
#include <cstring>
#include <thread>
#include <atomic>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef int socklen_t;
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#endif

namespace tenzo {
namespace net {

namespace {

bool sendAll(int sock, const void* data, size_t size) {
    const char* ptr = reinterpret_cast<const char*>(data);
    size_t remaining = size;
    while (remaining > 0) {
#ifdef _WIN32
        int sent = ::send(sock, ptr, (int)remaining, 0);
#else
        ssize_t sent = ::send(sock, ptr, remaining, 0);
#endif
        if (sent <= 0) return false;
        remaining -= sent;
        ptr += sent;
    }
    return true;
}

bool recvAll(int sock, void* data, size_t size) {
    char* ptr = reinterpret_cast<char*>(data);
    size_t remaining = size;
    while (remaining > 0) {
#ifdef _WIN32
        int received = ::recv(sock, ptr, (int)remaining, 0);
#else
        ssize_t received = ::recv(sock, ptr, remaining, 0);
#endif
        if (received <= 0) return false;
        remaining -= received;
        ptr += received;
    }
    return true;
}

void setNoDelay(int sock) {
    int flag = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<char*>(&flag), sizeof(flag));
}

} // namespace

TenzoClient::TenzoClient(const std::string& host, int port)
    : host(host), port(port), socketFd(-1) {}

TenzoClient::~TenzoClient() {
    disconnect();
}

bool TenzoClient::connect() {
    disconnect();

    socketFd = socket(AF_INET, SOCK_STREAM, 0);
    if (socketFd < 0) return false;

    setNoDelay(socketFd);

    struct sockaddr_in serv_addr;
    std::memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, host.c_str(), &serv_addr.sin_addr) <= 0) {
        disconnect();
        return false;
    }

    if (::connect(socketFd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        disconnect();
        return false;
    }

    return true;
}

void TenzoClient::disconnect() {
    if (socketFd >= 0) {
#ifdef _WIN32
        closesocket(socketFd);
#else
        close(socketFd);
#endif
        socketFd = -1;
    }
}

bool TenzoClient::isConnected() const {
    return socketFd >= 0;
}

double TenzoClient::ping() {
    if (!isConnected()) {
        if (!connect()) return -1.0;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    PacketHeader req{};
    req.magic = TENZO_NET_MAGIC;
    req.command = 3; // PING
    req.dataBytes = 0;

    if (!sendAll(socketFd, &req, sizeof(req))) {
        disconnect();
        return -1.0;
    }

    PacketHeader resp{};
    if (!recvAll(socketFd, &resp, sizeof(resp)) || resp.command != 4) {
        disconnect();
        return -1.0;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

bool TenzoClient::executeRemoteLayers(uint32_t seqId,
                                      uint16_t startLayer,
                                      uint16_t endLayer,
                                      const float* inActivation,
                                      size_t dim,
                                      float* outActivation) {
    if (!isConnected()) {
        if (!connect()) return false;
    }

    PacketHeader req;
    req.magic = TENZO_NET_MAGIC;
    req.command = 1; // EXEC_LAYERS
    req.seqId = seqId;
    req.startLayer = startLayer;
    req.endLayer = endLayer;
    req.dataBytes = static_cast<uint32_t>(dim * sizeof(float));

    if (!sendAll(socketFd, &req, sizeof(req))) {
        disconnect();
        return false;
    }

    if (!sendAll(socketFd, inActivation, req.dataBytes)) {
        disconnect();
        return false;
    }

    // Await response
    PacketHeader resp;
    if (!recvAll(socketFd, &resp, sizeof(resp))) {
        disconnect();
        return false;
    }

    if (resp.magic != TENZO_NET_MAGIC || resp.command != 2 || resp.dataBytes != req.dataBytes) {
        disconnect();
        return false;
    }

    if (!recvAll(socketFd, outActivation, resp.dataBytes)) {
        disconnect();
        return false;
    }

    return true;
}

TenzoServer::TenzoServer(int port, LayerExecCallback callback)
    : port(port), serverFd(-1), callback(callback), running(false) {}

TenzoServer::~TenzoServer() {
    stop();
}

bool TenzoServer::start() {
    serverFd = socket(AF_INET, SOCK_STREAM, 0);
    if (serverFd < 0) return false;

    int opt = 1;
    setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<char*>(&opt), sizeof(opt));

    struct sockaddr_in address;
    std::memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(serverFd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        stop();
        return false;
    }

    if (listen(serverFd, 5) < 0) {
        stop();
        return false;
    }

    running = true;
    listenThread = std::make_unique<std::thread>([this]() {
        while (running) {
            int currentFd = serverFd;
            if (currentFd < 0) break;

#ifdef _WIN32
            WSAPOLLFD pfd;
            pfd.fd = currentFd;
            pfd.events = POLLRDNORM;
            pfd.revents = 0;
            int pr = WSAPoll(&pfd, 1, 100);
#else
            struct pollfd pfd;
            pfd.fd = currentFd;
            pfd.events = POLLIN;
            pfd.revents = 0;
            int pr = poll(&pfd, 1, 100);
#endif
            if (!running) break;
            if (pr <= 0) continue;

            struct sockaddr_in clientAddr;
            socklen_t addrLen = sizeof(clientAddr);
            int clientFd = accept(currentFd, (struct sockaddr*)&clientAddr, &addrLen);
            if (clientFd < 0) {
                if (!running) break;
                continue;
            }
            setNoDelay(clientFd);
            std::thread(&TenzoServer::handleClient, this, clientFd).detach();
        }
    });

    return true;
}

void TenzoServer::stop() {
    running = false;
    if (serverFd >= 0) {
        int fd = serverFd;
        serverFd = -1;
#ifdef _WIN32
        shutdown(fd, SD_BOTH);
        closesocket(fd);
#else
        shutdown(fd, SHUT_RDWR);
        close(fd);
#endif
    }
    if (listenThread && listenThread->joinable()) {
        listenThread->join();
    }
}

void TenzoServer::handleClient(int clientFd) {
    std::vector<float> inBuf;
    std::vector<float> outBuf;

    while (running) {
        PacketHeader req;
        if (!recvAll(clientFd, &req, sizeof(req))) break;

        if (req.magic != TENZO_NET_MAGIC) break;

        if (req.command == 3) { // PING
            PacketHeader resp{};
            resp.magic = TENZO_NET_MAGIC;
            resp.command = 4; // PONG
            resp.seqId = req.seqId;
            resp.dataBytes = 0;
            if (!sendAll(clientFd, &resp, sizeof(resp))) break;
            continue;
        }

        size_t numFloats = req.dataBytes / sizeof(float);
        if (inBuf.size() < numFloats) {
            inBuf.resize(numFloats);
            outBuf.resize(numFloats);
        }

        if (!recvAll(clientFd, inBuf.data(), req.dataBytes)) break;

        bool ok = callback(req.startLayer, req.endLayer, inBuf.data(), numFloats, outBuf.data());

        PacketHeader resp;
        resp.magic = TENZO_NET_MAGIC;
        resp.command = ok ? 2 : 0xFF; // RESULT_ACTIVATION
        resp.seqId = req.seqId;
        resp.startLayer = req.startLayer;
        resp.endLayer = req.endLayer;
        resp.dataBytes = req.dataBytes;

        if (!sendAll(clientFd, &resp, sizeof(resp))) break;
        if (!sendAll(clientFd, outBuf.data(), resp.dataBytes)) break;
    }

#ifdef _WIN32
    closesocket(clientFd);
#else
    close(clientFd);
#endif
}

} // namespace net
} // namespace tenzo
