#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <new>

namespace ldpc {
namespace detail {

/// Uninitialised, 64-byte aligned heap buffer.
class AlignedBuffer {
public:
    static constexpr std::size_t kAlignment = 64;

    AlignedBuffer() = default;
    explicit AlignedBuffer(std::size_t bytes)
        : data_(bytes ? static_cast<unsigned char*>(::operator new(bytes, std::align_val_t{kAlignment})) : nullptr),
          size_(bytes) {}

    void* data() noexcept { return data_.get(); }
    std::size_t size() const noexcept { return size_; }

private:
    struct Deleter {
        void operator()(unsigned char* p) const noexcept { ::operator delete(p, std::align_val_t{kAlignment}); }
    };
    std::unique_ptr<unsigned char, Deleter> data_;
    std::size_t size_ = 0;
};

}  // namespace detail
}  // namespace ldpc
