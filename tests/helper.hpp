// SPDX-License-Identifier: BSD-3-Clause

#include <vector>
#include <iostream>

namespace tpack::util {

using Dimensions = std::vector< std::size_t >;

using PartitionLevel = std::vector< std::size_t >;
using Partition      = std::vector< PartitionLevel >;
using PartitionList  = std::vector< Partition >;

struct TensorInfo {
	Dimensions dims;
	PartitionList partitions;
};

inline std::ostream &operator<<(std::ostream &stream, const TensorInfo &info) {
	stream << "{ .dims={";

	for (std::size_t i = 0; i < info.dims.size(); ++i) {
		stream << info.dims[i];

		if (i + 1 < info.dims.size()) {
			stream << ", ";
		}
	}

	stream << "}, .partition={";
	for (std::size_t i = 0; i < info.partitions.size(); ++i) {
		stream << " {";

		for (std::size_t k = 0; k < info.partitions[i].size(); ++k) {
			stream << " { ";

			for (std::size_t m = 0; m < info.partitions[i][k].size(); ++m) {
				stream << info.partitions[i][k][m];

				if (m + 1 < info.partitions[i][k].size()) {
					stream << ", ";
				}
			}

			stream << " }";
		}

		stream << " }";
	}

	stream << " }";

	return stream;
}

inline TensorInfo make_info(std::vector< std::size_t > dims, PartitionList parts) {
	return { .dims = std::move(dims), .partitions = std::move(parts) };
}

inline TensorInfo make_info_p(std::vector< std::size_t > dims, Partition part) {
	return { .dims = std::move(dims), .partitions = PartitionList{ { std::move(part) } } };
}

inline TensorInfo make_info_l(std::vector< std::size_t > dims, PartitionLevel level) {
	return { .dims = std::move(dims), .partitions = PartitionList{ Partition{ std::move(level) } } };
}

} // namespace tpack::util
