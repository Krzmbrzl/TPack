// SPDX-License-Identifier: BSD-3-Clause

#pragma once

/**
 * @file tpack.hpp
 * @brief Umbrella header pulling in the entire public TPack API.
 *
 * Include this to get every public entry point at once. Translation units that only need part of the API may
 * instead include the individual headers (e.g. @c <tpack/rank.hpp>).
 */

#include <tpack/indexing.hpp>
#include <tpack/labeled.hpp>
#include <tpack/orbit.hpp>
#include <tpack/partition.hpp>
#include <tpack/rank.hpp>
#include <tpack/sector_layout.hpp>
#include <tpack/symmetry.hpp>
