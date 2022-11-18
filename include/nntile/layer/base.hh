/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/layer/base.hh
 * Base API for all layers
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-11-18
 * */

#pragma once

#include <nntile/tensor.hh>

namespace nntile
{
namespace layer
{

//! Common API for all layers
template<typename T>
class Base
{
public:
    tensor::TensorTraits input_traits, output_traits;
    std::vector<tensor::Tensor<T>> params;
    std::vector<tensor::Tensor<T>> grads;
    Base(const tensor::TensorTraits &input_traits_,
            const tensor::TensorTraits &output_traits_,
            const std::vector<tensor::Tensor<T>> &params_,
            const std::vector<tensor::Tensor<T>> &grads_):
        input_traits(input_traits_),
        output_traits(output_traits_),
        params(params_),
        grads(grads_)
    {
        // Check params and grads have the same sizes
        if(params.size() != grads.size())
        {
            throw std::runtime_error("params.size() != grads.size()");
        }
        for(Index i = 0; i < params.size(); ++i)
        {
            if(params[i].shape != grads[i].shape)
            {
                throw std::runtime_error("params[i].shape != "
                        "grads[i].shape");
            }
            if(params[i].basetile_shape != grads[i].basetile_shape)
            {
                throw std::runtime_error("params[i].basetile_shape != "
                        "grads[i].basetile_shape");
            }
        }
    }
    // Destructor is virtual since this is a base class for all layers
    virtual ~Base() = default;
    void forward_async(const tensor::Tensor<T> &input,
            const tensor::Tensor<T> &output) const = delete;
    void backward_async(const tensor::Tensor<T> &forward_input,
            const tensor::Tensor<T> &input,
            const tensor::Tensor<T> &output) const = delete;
    void grads_invalidate_submit() const
    {
        for(auto t: grads)
        {
            t.invalidate_submit();
        }
    }
};

} // namespace layer
} // namespace nntile

