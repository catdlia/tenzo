#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <future>
#include "context/HardwareInfo.h"

namespace tenzo {

/// A hardware-aware thread pool that supports core pinning (affinity)
class ThreadPool {
public:
    explicit ThreadPool(const TopologyInfo& topo, bool useAffinity = true);
    ~ThreadPool();

    /// Enqueue a task to be executed on a specific CPU
    /// If cpuId is -1, any available thread will pick it up
    template<class F, class... Args>
    auto enqueue(int cpuId, F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;

    /// Executes tasks based on HeterogeneousWorkSplit assignments
    void executeSplit(const HeterogeneousWorkSplit& split, 
                      std::function<void(const HeterogeneousWorkSplit::ThreadWork&)> task);

    int size() const { return (int)workers.size(); }

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

} // namespace tenzo
