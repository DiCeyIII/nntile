/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/softmax/cpu.hh
 * Softmax operation on CPU
 *
 * @version 1.0.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::softmax
{

template<typename T>
void cpu(Index m, Index n, Index k, const T *maxsumexp, const T *src, scal_t alpha,
        T *dst)
    noexcept;

} // namespace nntile::kernel::softmax
