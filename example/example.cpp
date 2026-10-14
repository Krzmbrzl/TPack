// SPDX-License-Identifier: BSD-3-Clause

#include <tpack/orbit.hpp>

#include <cstddef>
#include <iostream>
#include <vector>


int main() {
	const std::vector< std::size_t > dims                                     = { 5, 5 };
	const std::vector< std::vector< std::vector< std::size_t > > > partitions = { { { 0, 1 } } };

	std::cout << tpack::num_orbits(dims, partitions) << std::endl;

	return 0;
}
