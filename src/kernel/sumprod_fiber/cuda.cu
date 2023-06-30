/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sumprod_fiber/cuda.cu
 * Sums over slices into a fiber of a product of buffers on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-06-20
 * */

#include "nntile/kernel/sumprod_fiber/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace sumprod_fiber
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, T alpha, const T *src1,
        const T *src2, T beta, T *dst)
//! Sums over slices into a fiber of a product of two tensors on CPU
/*! For two provided m-by-k-by-n input arrays src1 and src2 compute sums of
 * per-element product of corresponding slices along the first and the third
 * axes with m and n elements respectively, resulting in output vector dst with
 * k elements.
 * Mnemonically, the following operations are performed:
 *      dst[l] = beta*dst[l] + alpha*sum_ij(src1[i,l,j] * src2[i,l,j])
 *      
 * @param[in] m: Size of the first mode of src1 and src2 tensors
 * @param[in] n: Size of the last mode of src1 and src2 tensors
 * @param[in] k: Size of the middle mode of src1 and src2 tensors and of the
 *      only mode of dst tensor
 * @param[in] alpha: Scaling factor for src1*src2
 * @param[in] src1: Input contiguous m-by-k-by-n array
 * @param[in] src2: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Output contiguous vector with k elements, that
 *      accumulates sums along the first and the last axes of per-element
 *      products of src1 and src2.
 * */
{
    Index i2 = threadIdx.x + blockIdx.x*blockDim.x;
    Index i0_start = threadIdx.y, i0_step = blockDim.y;
    Index i1_start = threadIdx.z, i1_step = blockDim.z;
    constexpr T zero = 0;
    // Init sum of product of the slices
    T sum = zero;
    if(i2 < k)
    {
        // Cycle over column of src1 and src2
        for(Index i1 = i1_start; i1 < n; i1 += i1_step)
        {
            // Get corresponding fibers of both sources
            const T *src1_fiber = src1 + (i1*k+i2)*m;
            const T *src2_fiber = src2 + (i1*k+i2)*m;
            // Cycle over fibers of inputs
            for(Index i0 = i0_start; i0 < m; i0 += i0_step)
            {
                // Update sum
                sum += src1_fiber[i0] * src2_fiber[i0];
            }
        }
    }
    __shared__ T block_sum[2];
    if(i1_start == 0 and i0_start == 0)
    {
        block_sum[threadIdx.x] = zero;
    }
    __syncthreads();
    atomicAdd(&block_sum[threadIdx.x], sum);
    __syncthreads();
    // Update output value
    if(i1_start == 0 and i0_start == 0 and i2 < k)
    {
        // Output value
        T &result = dst[i2];
        if(beta == zero)
        {
            result = alpha * block_sum[threadIdx.x];
        }
        else
        {
            result = beta*result + alpha*block_sum[threadIdx.x];
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, T alpha,
        const T *src1, const T *src2, T beta, T *dst)
    noexcept
//! Sums over slices into a fiber of a product of two tensors on CPU
/*! For two provided m-by-k-by-n input arrays src1 and src2 compute sums of
 * per-element product of corresponding slices along the first and the third
 * axes with m and n elements respectively, resulting in output vector dst with
 * k elements.
 * Mnemonically, the following operations are performed:
 *      dst[l] = beta*dst[l] + alpha*sum_ij(src1[i,l,j] * src2[i,l,j])
 *      
 * @param[in] m: Size of the first mode of src1 and src2 tensors
 * @param[in] n: Size of the last mode of src1 and src2 tensors
 * @param[in] k: Size of the middle mode of src1 and src2 tensors and of the
 *      only mode of dst tensor
 * @param[in] alpha: Scaling factor for src1*src2
 * @param[in] src1: Input contiguous m-by-k-by-n array
 * @param[in] src2: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Output contiguous vector with k elements, that
 *      accumulates sums along the first and the last axes of per-element
 *      products of src1 and src2.
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(2, std::min(int(m), 32), std::min(int(n), 32));
    dim3 blocks((k+threads.x-1)/threads.x, 1, 1);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, alpha, src1,
            src2, beta, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, fp32_t alpha,
        const fp32_t *src1, const fp32_t *src2, fp32_t beta, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, fp64_t alpha,
        const fp64_t *src1, const fp64_t *src2, fp64_t beta, fp64_t *dst)
    noexcept;

} // namespace sumprod_fiber
} // namespace kernel
} // namespace nntile

