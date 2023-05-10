/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/prod_slice.cc
 * Tile wrappers for per-element product of a tensor and a broadcasted slice
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-02
 * */

#include "nntile/tile/prod_slice.hh"
#include "nntile/starpu/prod_slice.hh"

namespace nntile
{
namespace tile
{

template<typename T>
void prod_slice_async(const Tile<T> &src, T alpha, const Tile<T> &dst,
        Index axis)
//! Tile<T> per-element multiplication of a tensor and a broadcasted slice
/*! Reshapes input tensor and slice into 3-dimensional and 2-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j] = alpha * dst[i,l,j] * src[i,j]
 *
 * @param[in] src: Input slice, that is reshaped into 2D array
 * @param[in] alpha: Scalar factor
 * @param[inout] dst: Resulting tensor, that is reshaped into 3D array
 * */
{
    // Check dimensions
    if(dst.ndim != src.ndim+1)
    {
        throw std::runtime_error("dst.ndim != src.ndim+1");
    }
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= dst.ndim)
    {
        throw std::runtime_error("axis >= dst.ndim");
    }
    // Check shapes of tiles
    for(Index i = 0; i < axis; ++i)
    {
        if(dst.shape[i] != src.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i]");
        }
    }
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src.shape[i-1])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i-1]");
        }
    }
    // Reshape inputs for simplicity: src -> (m,n), dst -> (m,k,n)
    Index m, n, k;
    m = dst.stride[axis];
    n = dst.matrix_shape[axis+1][1];
    k = dst.shape[axis];
    // Insert corresponding task
    starpu::prod_slice::submit<T>(m, n, k, alpha, src, dst);
}

template<typename T>
void prod_slice(const Tile<T> &src, T alpha, const Tile<T> &dst, Index axis)
//! Tile<T> per-element multiplication of a tensor and a broadcasted slice
/*! Blocking version of prod_slice_async<T>.
 * Reshapes input tensor and slice into 3-dimensional and 2-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j] = alpha * dst[i,l,j] * src[i,j]
 *
 * @param[in] src: Input slice, that is reshaped into 2D array
 * @param[in] alpha: Scalar factor
 * @param[inout] dst: Resulting tensor, that is reshaped into 3D array
 * */
{
    prod_slice_async<T>(src, alpha, dst, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void prod_slice_async<fp32_t>(const Tile<fp32_t> &src, fp32_t alpha,
        const Tile<fp32_t> &dst, Index axis);

template
void prod_slice_async<fp64_t>(const Tile<fp64_t> &src, fp64_t alpha,
        const Tile<fp64_t> &dst, Index axis);

// Explicit instantiation of template
template
void prod_slice<fp32_t>(const Tile<fp32_t> &src, fp32_t alpha,
        const Tile<fp32_t> &dst, Index axis);

template
void prod_slice<fp64_t>(const Tile<fp64_t> &src, fp64_t alpha,
        const Tile<fp64_t> &dst, Index axis);

} // namespace tile
} // namespace nntile

