/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sumprod_fiber/cpu.hh
 * Sums over slices into a fiber of a product of buffers on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-02
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace sumprod_fiber
{

template<typename T>
void cpu(Index m, Index n, Index k, T alpha, const T *src1, const T *src2,
        T beta, T *dst)
    noexcept;

} // namespace sumprod_fiber
} // namespace kernel
} // namespace nntile

