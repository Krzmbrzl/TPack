// SPDX-License-Identifier: BSD-3-Clause

#include <tpack/details/binomial.hpp>
#include <tpack/details/factorial.hpp>

#include <benchmark/benchmark.h>


static void BM_factorial(benchmark::State &state) {
	const std::size_t n = state.range();

	for (auto _ : state) {
		std::size_t result = tpack::details::factorial(n);
		benchmark::DoNotOptimize(result);
	}
}

BENCHMARK(BM_factorial)->DenseRange(0, 20);


static void BM_binomial(benchmark::State &state) {
	const std::size_t n = state.range(0);
	const std::size_t k = state.range(1);

	for (auto _ : state) {
		std::size_t result = tpack::details::binomial(n, k);
		benchmark::DoNotOptimize(result);
	}
}

BENCHMARK(BM_binomial)->ArgsProduct({ { 12, 24, 48, 96, 192, 384, 768 }, { 2, 4, 6, 8 } });
