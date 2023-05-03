/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gelu/cuda.cu
 * GeLU operation on a buffer on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-03
 * */

#include "nntile/kernel/gelu/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace gelu
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, T *data)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    constexpr T mone = -1, pt5 = 0.5;
    const T f1 = mone / sqrt(T{2.0});
    if(i < nelems)
    {
        T z = data[i];
        T y = erfc(f1 * z);
        data[i] = pt5 * z * y;
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, T *data)
    noexcept
//! Inplace GeLU operation performed on CUDA
/*! Uses very slow std::erfc() function, so consider using approximated version
 * nntile::kernel::cpu::gelutanh(). Does the following per-element operation:
 * GeLU(z) = 0.5 z erfc(-z/sqrt(2))
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[inout] data: Buffer to apply GeLU
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, data);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, fp32_t *data)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, fp64_t *data)
    noexcept;

} // namespace gelu
} // namespace kernel
} // namespace nntile

