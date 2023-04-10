/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/gelu_backward.hh
 * Backward GeLU low-level kernels
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-04-05
 * */

#pragma once

#include <nntile/kernel/gelu_backward/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/gelu_backward/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::gelu_backward
/*! Low-level implementations of backward GeLU operation
 * */
namespace gelu_backward
{

} // namespace gelu_backward
} // namespace kernel
} // namespace nntile

