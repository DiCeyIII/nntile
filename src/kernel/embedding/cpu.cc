/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/embedding/cpu.cc
 * Embeddings from vocabulary within buffers
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-21
 * */

#include "nntile/kernel/embedding/cpu.hh"

namespace nntile
{
namespace kernel
{
namespace embedding
{

template<typename T>
void cpu(Index m, Index n, Index k, Index k_start, Index k_size,
        const Index *index, const T *vocab, T *embed)
    noexcept
//! Fill embedding from vocabulary
/*! Fill provided m-by-k-by-n output tensor embed:
 *      embed[i, k_start:k_start+k_size, j] = vocab[:, index[i, j]]
 *
 * @param[in] m: Size of the first mode of index and embed tensors
 * @param[in] n: Size of the last mode of index and embed tensors
 * @param[in] k: Size of the middle mode of embed tensor
 * @param[in] k_start: Offset of the middle mode of embed tensor
 * @param[in] k_size: Size of the first mode of vocab tensor
 * @param[in] index: Tokens (indices of embeddings)
 * @param[in] vocab: Vocabulary of embeddings. It is a contiguous matrix of shape
 *      (k_size, vocab_size) but vocab_size is not passed as a parameter.
 * @param[inout] embed: Output tensor to be filled with embeddings
 * */
{
    // Cycle over column of output buffer
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Cycle over row of output buffer
        for(Index i1 = 0; i1 < m; ++i1)
        {
            // Input slice of vocabulary
            const T *vocab_slice = vocab + k_size*index[i2*m+i1];
            // Output slice to be updated
            T *embed_slice = embed + (i2*k+k_start)*m + i1;
            // Cycle over slice over middle axis of output buffer
            for(Index i0 = 0; i0 < k_size; ++i0)
            {
                embed_slice[i0*m] = vocab_slice[i0];
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, Index k_start, Index k_size,
        const Index *index, const fp32_t *vocab, fp32_t *embed)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, Index k_start, Index k_size,
        const Index *index, const fp64_t *vocab, fp64_t *embed)
    noexcept;

} // namespace embedding
} // namespace kernel
} // namespace nntile

