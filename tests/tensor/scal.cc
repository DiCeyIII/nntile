/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/scal.cc
 * SCAL operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-03-29
 * */

#include "nntile/tensor/scal.hh"
#include "nntile/tile/scal.hh"
#include "nntile/starpu/scal.hh"
#include "nntile/tensor/scatter.hh"
#include "nntile/tensor/gather.hh"
#include "nntile/starpu/subcopy.hh"
#include "../testing.hh"
#include <limits>

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void check(T alpha, const std::vector<Index> &shape,
        const std::vector<Index> &basetile)
{
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Some preparation
    starpu_mpi_tag_t last_tag = 0;
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    int mpi_root = 0;
    // Generate single-tile source tensor and init it
    TensorTraits data_single_traits(shape, shape);
    std::vector<int> dist_root = {mpi_root};
    Tensor<T> data_single(data_single_traits, dist_root, last_tag);
    Tensor<T> data2_single(data_single_traits, dist_root, last_tag);
    if(mpi_rank == mpi_root)
    {
        auto tile = data_single.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(Index i = 0; i < data_single.nelems; ++i)
        {
            tile_local[i] = T(i);
        }
        tile_local.release();
    }
    // Scatter source tensor
    TensorTraits data_traits(shape, basetile);
    std::vector<int> data_distr(data_traits.grid.nelems);
    for(Index i = 0; i < data_traits.grid.nelems; ++i)
    {
        data_distr[i] = (i+1) % mpi_size;
    }
    Tensor<T> data(data_traits, data_distr, last_tag);
    scatter<T>(data_single, data);
    // Perform tensor-wise and tile-wise scal operations
    scal<T>(alpha, data);
    if(mpi_rank == mpi_root)
    {
        tile::scal<T>(alpha, data_single.get_tile(0));
    }
    gather<T>(data, data2_single);
    // Compare results
    if(mpi_rank == mpi_root)
    {
        auto tile = data_single.get_tile(0);
        auto tile2 = data2_single.get_tile(0);
        auto tile_local = tile.acquire(STARPU_R);
        auto tile2_local = tile2.acquire(STARPU_R);
        for(Index i = 0; i < data_single.nelems; ++i)
        {
            TEST_ASSERT(tile_local[i] == tile2_local[i]);
        }
        tile_local.release();
        tile2_local.release();
    }
}

template<typename T>
void validate()
{
    check<T>(2.0, {11}, {5});
    check<T>(-1.0, {11, 12}, {5, 6});
    check<T>(1.0, {11, 12, 13}, {5, 6, 5});
    // Sync to guarantee old data tags are cleaned up and can be reused
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Check throwing exceptions
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::scal::init();
    starpu::subcopy::init();
    starpu::scal::restrict_where(STARPU_CPU);
    starpu::subcopy::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}

