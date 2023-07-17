#include <cuda.h>
#include <nvbench/launch.cuh>
#include <nvbench/nvbench.cuh>

#include "nntile/base_types.hh"
#include "nntile/kernel/maxsumexp.hh"
#include "nntile/kernel/maxsumexp/cuda.hh"

namespace maxsumexp = nntile::kernel::maxsumexp;

enum class Device : int {
    kCPU = 0,
    kCUDA = 1,
};

template <typename T, Device device> struct Array;

template <typename T> struct Array<T, Device::kCUDA> {
    size_t size;
    T *data = nullptr;
    cudaError_t status = cudaSuccess;

    Array(size_t size) noexcept : size{size} {
        status = cudaMalloc(&data, size * sizeof(T));
    }

    ~Array(void) {
        if (data) {
            cudaFree(data);
            data = nullptr;
        }
    }

    operator bool(void) const {
        return status == cudaError_t::cudaSuccess;
    }

    template <typename U> U *as(void) {
        return reinterpret_cast<U *>(data);
    }

    template <typename U> U *as(void) const {
        return reinterpret_cast<U const *>(data);
    }
};

void BenchMaxSumExp(nvbench::state &state) {
    auto batch_size = static_cast<int>(state.get_int64("batch_size"));
    auto seq_len = static_cast<int>(state.get_int64("seq_len"));
    auto src = Array<float, Device::kCUDA>(batch_size * seq_len * seq_len);
    auto dst = Array<float, Device::kCUDA>(batch_size * seq_len * 2);

    // Request throughput stats.
    state.add_element_count(src.size);
    state.add_global_memory_reads<float>(src.size);
    state.add_global_memory_writes<float>(dst.size);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
        maxsumexp::cuda(launch.get_stream(), batch_size, seq_len, seq_len,
                        src.as<float>(), dst.as<float>());
        cudaStreamSynchronize(launch.get_stream());
    });
}

NVBENCH_BENCH(BenchMaxSumExp)
    .add_int64_axis("batch_size", {2, 8, 32})
    .add_int64_axis("seq_len", {64, 256});
