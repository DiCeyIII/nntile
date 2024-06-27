/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sum_fiber/cpu.hh
 * Sums over slices into a fiber of a buffer on CPU
 *
 * @version 1.0.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::sum_fiber
{

// Sums over slices along the first and last axes into a fiber of a tensor
template<typename T>
void cpu(Index m, Index n, Index k, Index batch, scal_t alpha, const T *src, scal_t beta,
        T *dst)
    noexcept;

} // namespace nntile::kernel::sum_fiber
