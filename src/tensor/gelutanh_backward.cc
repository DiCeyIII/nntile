/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/gelutanh_backward.cc
 * Backward approximate GeLU operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-04-05
 * */

#include "nntile/tensor/gelutanh_backward.hh"
#include "nntile/starpu/gelutanh_backward.hh"

namespace nntile
{
namespace tensor
{

//! Asynchronous tensor-wise backward approximate GeLU operation
//
// @param[inout] A: Tensor for the element-wise backward approximate GeLU operation
template<typename T>
void gelutanh_backward_async(const Tensor<T> &x, const Tensor<T> &dy,
        const Tensor<T> &dx)
{
    // Check shapes
    if(x.shape != dy.shape)
    {
        throw std::runtime_error("x.shape != dy.shape");
    }
    if(x.basetile_shape != dy.basetile_shape)
    {
        throw std::runtime_error("x.basetile_shape != dy.basetile_shape");
    }
    if(x.shape != dx.shape)
    {
        throw std::runtime_error("x.shape != dx.shape");
    }
    if(x.basetile_shape != dx.basetile_shape)
    {
        throw std::runtime_error("x.basetile_shape != dx.basetile_shape");
    }
    // Do actual calculations
    int mpi_rank = starpu_mpi_world_rank();
    for(Index i = 0; i < x.grid.nelems; ++i)
    {
        auto x_tile_handle = x.get_tile_handle(i);
        auto dy_tile_handle = dy.get_tile_handle(i);
        auto dx_tile_handle = dx.get_tile_handle(i);
        // Execution node
        int exec_rank = dx_tile_handle.mpi_get_rank();
        // Execution node submission
        if(mpi_rank == exec_rank)
        {
            auto x_tile_traits = x.get_tile_traits(i);
            starpu::gelutanh_backward::submit_mpi<T>(x_tile_traits.nelems,
                    x_tile_handle, dy_tile_handle, dx_tile_handle, exec_rank);
        }
        // MPI transfers submission
        else if(mpi_rank == x_tile_handle.mpi_get_rank()
                or mpi_rank == dy_tile_handle.mpi_get_rank()
                or mpi_rank == dx_tile_handle.mpi_get_rank())
        {
            starpu::gelutanh_backward::submit_mpi<T>(0,
                    x_tile_handle, dy_tile_handle, dx_tile_handle, exec_rank);
        }
        // Clear cached output value
        dx_tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise backward approximate GeLU operation
//
// @param[inout] A: Tensor for the element-wise backward approximate GeLU operation
template<typename T>
void gelutanh_backward(const Tensor<T> &x, const Tensor<T> &dy,
        const Tensor<T> &dx)
{
    gelutanh_backward_async<T>(x, dy, dx);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void gelutanh_backward_async<fp32_t>(const Tensor<fp32_t> &x,
        const Tensor<fp32_t> &dy, const Tensor<fp32_t> &dx);

template
void gelutanh_backward_async<fp64_t>(const Tensor<fp64_t> &x,
        const Tensor<fp64_t> &dy, const Tensor<fp64_t> &dx);

// Explicit instantiation
template
void gelutanh_backward<fp32_t>(const Tensor<fp32_t> &x,
        const Tensor<fp32_t> &dy, const Tensor<fp32_t> &dx);

template
void gelutanh_backward<fp64_t>(const Tensor<fp64_t> &x,
        const Tensor<fp64_t> &dy, const Tensor<fp64_t> &dx);

} // namespace tensor
} // namespace nntile
