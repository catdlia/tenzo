#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <future>
#include <algorithm>
#include "context/HardwareProfile.h"

namespace tenzo {

/// High-performance, zero-dependency hardware-aware thread pool
class ThreadPool {
public:
    /// Construct with explicit thread count (0 = std::thread::hardware_concurrency)
    explicit ThreadPool(int numThreads = 0, bool useAffinity = false);

    /// Construct with TopologyInfo
    explicit ThreadPool(const TopologyInfo& topo, bool useAffinity = true);

    /// Construct with MLIR context TopologyInfo (if available)
    explicit ThreadPool(const std::vector<int>& cpuIds, bool useAffinity = true);

    ~ThreadPool();

    /// Global singleton instance for runtime operations
    static ThreadPool& getGlobalPool(int numThreads = 0);

    /// Enqueue a task to be executed
    template<class F, class... Args>
    auto enqueue(int cpuId, F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;

    /// Parallel for-loop with static chunking
    void parallelFor(int start, int end, int grainSize,
                     std::function<void(int, int)> body);

    /// Parallel for-loop with dynamic work stealing / chunking
    void parallelForDynamic(int start, int end, int chunkSize,
                            std::function<void(int, int)> body);

    /// Executes tasks based on HeterogeneousWorkSplit assignments
    void executeSplit(const HeterogeneousWorkSplit& split, 
                      std::function<void(const HeterogeneousWorkSplit::ThreadWork&)> task);

    /// Point-wise parallel for
    void parallelForItems(int start, int end, std::function<void(int)> body);

    int size() const { return (int)workers.size(); }
    size_t activeThreadCount() const { return workers.size(); }

private:
    struct Task {
        int preferredCpuId;
        std::function<void()> func;
    };

    std::vector<std::thread> workers;
    std::queue<Task> tasks;
    
    std::mutex queueMutex;
    std::condition_variable condition;
    std::atomic<bool> stop;

    // Internal worker function
    void workerLoop(int threadId, int cpuId, bool useAffinity);
};

// Implementation of template enqueue
template<class F, class... Args>
auto ThreadPool::enqueue(int cpuId, F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        if(stop) throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.push({cpuId, [task](){ (*task)(); }});
    }
    condition.notify_one();
    return res;
}

/// Convenience free function for parallel_for
template<typename Func>
inline void parallel_for(int start, int end, int grainSize, Func&& body) {
    ThreadPool::getGlobalPool().parallelFor(start, end, grainSize, std::forward<Func>(body));
}

/// Convenience free function for dynamic parallel_for
template<typename Func>
inline void parallel_for_dynamic(int start, int end, int chunkSize, Func&& body) {
    ThreadPool::getGlobalPool().parallelForDynamic(start, end, chunkSize, std::forward<Func>(body));
}

} // namespace tenzo
