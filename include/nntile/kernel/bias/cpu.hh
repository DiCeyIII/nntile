/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/bias/cpu.hh
 * Bias operation on a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-03-26
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace bias
{

// Apply bias along middle axis on CPU
template<typename T>
void cpu(Index m, Index n, Index k, T alpha, const T *src, T *dst)
    noexcept;

} // namespace bias
} // namespace kernel
} // namespace nntile

