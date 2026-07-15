// SPDX-License-Identifier: BSD-3-Clause

#include <tpack/orbit.hpp>

#include <cstddef>
#include <vector>


int main() {
	const std::vector< std::size_t > dims                                     = { 5, 5 };
	const std::vector< std::vector< std::vector< std::size_t > > > partitions = { { { 0, 1 } } };

	return tpack::num_orbits(dims, partitions);
}
