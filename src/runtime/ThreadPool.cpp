#include "ThreadPool.h"
#include <iostream>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace tenzo {

/// Pin the current thread to a specific CPU
static bool setThreadAffinity(int cpuId) {
    if (cpuId < 0) return false;
    
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpuId, &cpuset);
    
    pthread_t current = pthread_self();
    int result = pthread_setaffinity_np(current, sizeof(cpu_set_t), &cpuset);
    return result == 0;
#else
    return false;
#endif
}

ThreadPool::ThreadPool(int numThreads, bool useAffinity) : stop(false) {
    int count = numThreads > 0 ? numThreads : (int)std::thread::hardware_concurrency();
    if (count <= 0) count = 1;

    workers.reserve(count);
    for (int i = 0; i < count; i++) {
        workers.emplace_back([this, i, useAffinity] { 
            workerLoop(i, useAffinity ? i : -1, useAffinity); 
        });
    }
}

ThreadPool::ThreadPool(const TopologyInfo& topo, bool useAffinity) : stop(false) {
    auto cpuIds = topo.getAllPrimaryCpuIds();
    if (cpuIds.empty()) {
        int count = (int)std::thread::hardware_concurrency();
        if (count <= 0) count = 1;
        for (int i = 0; i < count; i++) {
            workers.emplace_back([this, i] { workerLoop(i, -1, false); });
        }
    } else {
        workers.reserve(cpuIds.size());
        for (size_t i = 0; i < cpuIds.size(); i++) {
            int cpuId = cpuIds[i];
            workers.emplace_back([this, i, cpuId, useAffinity] {
                workerLoop((int)i, cpuId, useAffinity);
            });
        }
    }
}

ThreadPool::ThreadPool(const std::vector<int>& cpuIds, bool useAffinity) : stop(false) {
    if (cpuIds.empty()) {
        int count = (int)std::thread::hardware_concurrency();
        if (count <= 0) count = 1;
        for (int i = 0; i < count; i++) {
            workers.emplace_back([this, i] { workerLoop(i, -1, false); });
        }
    } else {
        for (int i = 0; i < (int)cpuIds.size(); i++) {
            int cpuId = cpuIds[i];
            workers.emplace_back([this, i, cpuId, useAffinity] { workerLoop(i, cpuId, useAffinity); });
        }
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    condition.notify_all();
    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }
}

ThreadPool& ThreadPool::getGlobalPool(int numThreads) {
    static ThreadPool instance(numThreads, false);
    return instance;
}

void ThreadPool::workerLoop(int threadId, int cpuId, bool useAffinity) {
    if (useAffinity && cpuId >= 0) {
        setThreadAffinity(cpuId);
    }

    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            condition.wait(lock, [this] { return stop || !tasks.empty(); });
            
            if (stop && tasks.empty()) return;
            
            task = std::move(tasks.front());
            tasks.pop();
        }
        
        task.func();
    }
}

void ThreadPool::executeSplit(const HeterogeneousWorkSplit& split, 
                              std::function<void(const HeterogeneousWorkSplit::ThreadWork&)> task) {
    std::vector<std::future<void>> futures;
    for (const auto& work : split.assignments) {
        futures.push_back(enqueue(work.cpuId, [task, work]() {
            task(work);
        }));
    }
    for (auto& f : futures) {
        f.get();
    }
}

void ThreadPool::parallelFor(int start, int end, int grainSize,
                             std::function<void(int, int)> body) {
    if (start >= end) return;
    
    int totalWork = end - start;
    int numThreads = (int)workers.size();
    
    if (numThreads <= 1 || totalWork <= grainSize) {
        body(start, end);
        return;
    }
    
    int rawChunkSize = (totalWork + numThreads - 1) / numThreads;
    int chunkSize = grainSize > 0 ? ((rawChunkSize + grainSize - 1) / grainSize) * grainSize : rawChunkSize;
    if (chunkSize < 1) chunkSize = 1;
    
    std::vector<std::future<void>> futures;
    int offset = start;
    
    while (offset < end) {
        int chunkEnd = std::min(offset + chunkSize, end);
        int capturedStart = offset;
        int capturedEnd = chunkEnd;
        
        if (chunkEnd == end) {
            body(capturedStart, capturedEnd);
        } else {
            futures.push_back(enqueue(-1, [body, capturedStart, capturedEnd]() {
                body(capturedStart, capturedEnd);
            }));
        }
        
        offset = chunkEnd;
    }
    
    for (auto& f : futures) {
        f.get();
    }
}

void ThreadPool::parallelForDynamic(int start, int end, int chunkSize,
                                    std::function<void(int, int)> body) {
    if (start >= end) return;
    if (chunkSize <= 0) chunkSize = 1;

    std::atomic<int> nextWork(start);
    int numThreads = (int)workers.size();

    if (numThreads <= 1) {
        body(start, end);
        return;
    }

    auto workerTask = [&nextWork, end, chunkSize, &body]() {
        while (true) {
            int currentStart = nextWork.fetch_add(chunkSize, std::memory_order_relaxed);
            if (currentStart >= end) break;
            int currentEnd = std::min(currentStart + chunkSize, end);
            body(currentStart, currentEnd);
        }
    };

    std::vector<std::future<void>> futures;
    futures.reserve(numThreads - 1);
    for (int i = 0; i < numThreads - 1; ++i) {
        futures.push_back(enqueue(-1, workerTask));
    }

    // Main thread also participates
    workerTask();

    for (auto& f : futures) {
        f.get();
    }
}

void ThreadPool::parallelForItems(int start, int end, std::function<void(int)> body) {
    parallelFor(start, end, 1, [&body](int s, int e) {
        for (int i = s; i < e; ++i) {
            body(i);
        }
    });
}

} // namespace tenzo
