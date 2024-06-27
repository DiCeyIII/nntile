/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/add_slice3/cpu.hh
 * Per-element addition of a tensor and a broadcasted slice on CPU
 *
 * @version 1.0.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::add_slice3
{

// Per-element addition of a tensor and a broadcasted slice on CPU
template<typename T>
void cpu(Index m, Index n, Index k, scal_t alpha, const T *src1, scal_t beta,
        const T *src2, T *dst)
    noexcept;

} // namespace nntile::kernel::add_slice3
