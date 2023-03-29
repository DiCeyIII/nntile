/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel.hh
 * General info about namespace nntile::kernel
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Aleksandr Katrutsa
 * @author Konstantin Sozykin
 * @date 2023-03-27
 * */

#pragma once

#include <nntile/kernel/bias.hh>
#include <nntile/kernel/gelu.hh>
#include <nntile/kernel/gelutanh.hh>
#include <nntile/kernel/dgelu.hh>
#include <nntile/kernel/dgelutanh.hh>
#include <nntile/kernel/drelu.hh>
#include <nntile/kernel/hypot.hh>
#include <nntile/kernel/normalize.hh>
#include <nntile/kernel/prod.hh>
#include <nntile/kernel/randn.hh>
#include <nntile/kernel/relu.hh>
#include <nntile/kernel/subcopy.hh>
#include <nntile/kernel/sumnorm.hh>
#include <nntile/kernel/sum.hh>
#include <nntile/kernel/maxsumexp.hh>
#include <nntile/kernel/softmax.hh>
#include <nntile/kernel/sqrt.hh>
#include <nntile/kernel/maximum.hh>
#include <nntile/kernel/addcdiv.hh>
#include <nntile/kernel/scalprod.hh>
#include <nntile/kernel/logsumexp.hh>
#include <nntile/kernel/total_sum_accum.hh>
#include <nntile/kernel/subtract_indexed_column.hh>

namespace nntile
{
//! @namespace nntile::kernel
/*! This namespace holds low-level routines for codelets
 * */
namespace kernel
{

} // namespace kernel
} // namespace nntile

