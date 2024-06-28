/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/relu_backward.hh
 * Backward ReLU operation for Tensor<T>
 *
 * @version 1.0.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

template<typename T>
void relu_backward_async(const Tensor<T> &x, const Tensor<T> &dy,
        const Tensor<T> &dx);

template<typename T>
void relu_backward(const Tensor<T> &x, const Tensor<T> &dy,
        const Tensor<T> &dx);

} // namespace nntile::tensor
