#include "ThreadPool.h"
#include "llvm/Support/raw_ostream.h"

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
    // Windows or other OS: TODO implementation
    return false;
#endif
}

ThreadPool::ThreadPool(const TopologyInfo& topo, bool useAffinity) : stop(false) {
    auto cpuIds = topo.getAllPrimaryCpuIds();
    
    if (cpuIds.empty()) {
        int count = (int)std::thread::hardware_concurrency();
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

// Worker loop: pins thread to CPU and processes tasks from queue
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
            
            // For now, we use a simple queue. 
            // Future refinement: separate queues per core/type to ensure P-tasks go to P-cores.
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
        // We pass work.cpuId to enqueue to hint which thread should pick it up
        // Currently worker threads are fixed to cores, so we need a mechanism 
        // to match work.cpuId with thread-level affinity.
        
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
    
    // Single-chunk fast path
    if (numThreads <= 1 || totalWork <= grainSize) {
        body(start, end);
        return;
    }
    
    // Calculate chunk size aligned to grainSize
    int rawChunkSize = totalWork / numThreads;
    int chunkSize = (rawChunkSize / grainSize) * grainSize;
    if (chunkSize < grainSize) chunkSize = grainSize;
    
    std::vector<std::future<void>> futures;
    int offset = start;
    
    while (offset < end) {
        int chunkEnd = offset + chunkSize;
        if (chunkEnd > end || (end - chunkEnd) < grainSize) {
            // Give remainder to the last chunk to avoid tiny tail chunks
            chunkEnd = end;
        }
        
        int capturedStart = offset;
        int capturedEnd = chunkEnd;
        
        if (chunkEnd == end) {
            // Last chunk: run inline on calling thread to avoid queue overhead
            body(capturedStart, capturedEnd);
        } else {
            futures.push_back(enqueue(-1, [body, capturedStart, capturedEnd]() {
                body(capturedStart, capturedEnd);
            }));
        }
        
        offset = chunkEnd;
    }
    
    // Wait for all worker chunks to complete
    for (auto& f : futures) {
        f.get();
    }
}

} // namespace tenzo

