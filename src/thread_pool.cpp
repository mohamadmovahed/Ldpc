#include "thread_pool.hpp"

namespace ldpc {
namespace detail {

ThreadPool::ThreadPool(unsigned size) {
    const unsigned extra = size > 1 ? size - 1 : 0;
    threads_.reserve(extra);
    for (unsigned w = 1; w <= extra; ++w) {
        threads_.emplace_back([this, w] { worker_loop(w); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    start_cv_.notify_all();
    for (auto& t : threads_) {
        t.join();
    }
}

void ThreadPool::parallel_for(std::size_t num_tasks, const Task& task) {
    if (num_tasks == 0) {
        return;
    }
    if (threads_.empty() || num_tasks == 1) {
        for (std::size_t i = 0; i < num_tasks; ++i) {
            task(i, 0);
        }
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        task_ = &task;
        num_tasks_ = num_tasks;
        next_task_.store(0, std::memory_order_relaxed);
        busy_workers_ = static_cast<unsigned>(threads_.size());
        error_ = nullptr;
        ++generation_;
    }
    start_cv_.notify_all();
    run_tasks(0);

    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [this] { return busy_workers_ == 0; });
    task_ = nullptr;
    if (error_) {
        std::rethrow_exception(error_);
    }
}

void ThreadPool::worker_loop(unsigned worker) {
    std::uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            start_cv_.wait(lock, [&] { return stopping_ || generation_ != seen; });
            if (stopping_) {
                return;
            }
            seen = generation_;
        }
        run_tasks(worker);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            --busy_workers_;
        }
        done_cv_.notify_one();
    }
}

void ThreadPool::run_tasks(unsigned worker) noexcept {
    for (;;) {
        const std::size_t i = next_task_.fetch_add(1, std::memory_order_relaxed);
        if (i >= num_tasks_) {
            return;
        }
        try {
            (*task_)(i, worker);
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!error_) {
                error_ = std::current_exception();
            }
        }
    }
}

}  // namespace detail
}  // namespace ldpc
