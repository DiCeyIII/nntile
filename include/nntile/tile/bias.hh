/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/bias.hh
 * Bias operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-03-26
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

// Tile-wise bias operation
template<typename T>
void bias_async(T alpha, const Tile<T> &src, const Tile<T> &dst, Index axis);

// Tile-wise bias operation
template<typename T>
void bias(T alpha, const Tile<T> &src, const Tile<T> &dst, Index axis);

} // namespace tile
} // namespace nntile

