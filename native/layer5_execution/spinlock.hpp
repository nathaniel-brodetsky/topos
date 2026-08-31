#pragma once

#include <atomic>

class Spinlock {
public:
    void lock() {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            #if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();
            #endif
        }
    }

    void unlock() {
        flag_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

class SpinlockGuard {
public:
    explicit SpinlockGuard(Spinlock& lock) : lock_(lock) {
        lock_.lock();
    }
    ~SpinlockGuard() {
        lock_.unlock();
    }
    SpinlockGuard(const SpinlockGuard&) = delete;
    SpinlockGuard& operator=(const SpinlockGuard&) = delete;

private:
    Spinlock& lock_;
};
