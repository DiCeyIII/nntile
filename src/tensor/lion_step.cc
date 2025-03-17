/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/lion_step.cc
 * Fuse Lion step operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/lion_step.hh"
#include "nntile/starpu/lion_step.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise fused Lion step
template<typename T>
void lion_step_async(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar lambda, Scalar lr, Scalar weight_decay,
                     const Tensor<T> &grad, const Tensor<T> &first_moment, const Tensor<T> &p)
{
    if (p.matrix_shape != grad.matrix_shape)
    {
        throw std::runtime_error("Parameter shape is not equal to gradient shape");
    }
    if (p.matrix_shape != first_moment.matrix_shape)
    {
        throw std::runtime_error("Parameter shape is not equal to first_moment shape");
    }

    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();

    for (Index i = 0; i < p.grid.nelems; ++i)
    {
        // Get handle for corresponding tiles of tensors
        auto p_tile_handle            = p.get_tile_handle(i);
        auto grad_tile_handle         = grad.get_tile_handle(i);
        auto first_moment_tile_handle = first_moment.get_tile_handle(i);
        
        // Get MPI ranks of the corresponding tiles
        int p_tile_rank            = p_tile_handle.mpi_get_rank();
        int grad_tile_rank         = grad_tile_handle.mpi_get_rank();
        int first_moment_tile_rank = first_moment_tile_handle.mpi_get_rank();

        // Transfer data to the destination tile
        grad_tile_handle.mpi_transfer(p_tile_rank, mpi_rank);
        first_moment_tile_handle.mpi_transfer(p_tile_rank, mpi_rank);

        // Execute only on destination node
        if (mpi_rank == p_tile_rank)
        {
            auto traits = p.get_tile_traits(i);
            starpu::lion_step::submit<T>(num_iter, traits.nelems, beta_1, beta_2, lambda, lr, weight_decay,
                                         grad_tile_handle, first_moment_tile_handle, p_tile_handle);
        }
        // Flush cache for the output tile on every node
        p_tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise Lion step
template<typename T>
void lion_step(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar lambda, Scalar lr, Scalar weight_decay,
               const Tensor<T> &grad, const Tensor<T> &first_moment, const Tensor<T> &p)
{
    lion_step_async<T>(num_iter, beta_1, beta_2, lambda, lr, weight_decay, grad, first_moment, p);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void lion_step_async<fp32_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar lambda, Scalar lr, Scalar weight_decay,
    const Tensor<fp32_t> &grad, const Tensor<fp32_t> &first_moment, const Tensor<fp32_t> &p);

template
void lion_step_async<fp32_fast_tf32_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar lambda, Scalar lr, Scalar weight_decay,
    const Tensor<fp32_fast_tf32_t> &grad, const Tensor<fp32_fast_tf32_t> &first_moment, const Tensor<fp32_fast_tf32_t> &p);

template
void lion_step_async<fp64_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar lambda, Scalar lr, Scalar weight_decay,
    const Tensor<fp64_t> &grad, const Tensor<fp64_t> &first_moment, const Tensor<fp64_t> &p);

template
void lion_step_async<bf16_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar lambda, Scalar lr, Scalar weight_decay,
    const Tensor<bf16_t> &grad, const Tensor<bf16_t> &first_moment, const Tensor<bf16_t> &p);

// Explicit instantiation
template
void lion_step<fp32_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar lambda, Scalar lr, Scalar weight_decay,
    const Tensor<fp32_t> &grad, const Tensor<fp32_t> &first_moment, const Tensor<fp32_t> &p);

template
void lion_step<fp32_fast_tf32_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar lambda, Scalar lr, Scalar weight_decay,
    const Tensor<fp32_fast_tf32_t> &grad, const Tensor<fp32_fast_tf32_t> &first_moment, const Tensor<fp32_fast_tf32_t> &p);

template
void lion_step<fp64_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar lambda, Scalar lr, Scalar weight_decay,
    const Tensor<fp64_t> &grad, const Tensor<fp64_t> &first_moment, const Tensor<fp64_t> &p);

template
void lion_step<bf16_t>(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar lambda, Scalar lr, Scalar weight_decay,
    const Tensor<bf16_t> &grad, const Tensor<bf16_t> &first_moment, const Tensor<bf16_t> &p);

} // namespace nntile::tensor
