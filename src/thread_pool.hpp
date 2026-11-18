#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace ldpc {
namespace detail {

/// Fixed-size pool of persistent workers for fork-join loops. The calling
/// thread participates as worker 0, so a pool of size N spawns N-1 threads.
class ThreadPool {
public:
    using Task = std::function<void(std::size_t task, unsigned worker)>;

    explicit ThreadPool(unsigned size);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    unsigned size() const noexcept { return static_cast<unsigned>(threads_.size()) + 1; }

    /// Runs task(i, worker) for every i in [0, num_tasks) and blocks until all
    /// complete. Rethrows the first exception thrown by a task.
    void parallel_for(std::size_t num_tasks, const Task& task);

private:
    void worker_loop(unsigned worker);
    void run_tasks(unsigned worker) noexcept;

    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    const Task* task_ = nullptr;
    std::size_t num_tasks_ = 0;
    std::atomic<std::size_t> next_task_{0};
    unsigned busy_workers_ = 0;
    std::uint64_t generation_ = 0;
    bool stopping_ = false;
    std::exception_ptr error_;
};

}  // namespace detail
}  // namespace ldpc
