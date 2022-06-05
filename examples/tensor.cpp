#include <dllib/tensor.hpp>

#include <iostream>
#include <random>
#include <sys/resource.h>

static struct {
  struct StackSetter {
    StackSetter() {
      rlimit lim{};
      assert(getrlimit(RLIMIT_STACK, &lim) == 0);
      lim.rlim_cur = lim.rlim_max;
      std::cout << "New stack limit: " << lim.rlim_cur << std::endl;
      assert(setrlimit(RLIMIT_STACK, &lim) == 0);
    }
  } s{};
} __kek;

using namespace dllib;

template<class TFloat, size_t N>
size_t SingleRun() {
  TTensor<TFloat, N, N> a, b;
  std::mt19937 rnd(std::random_device{}());
  std::normal_distribution<TFloat> dist;
  for (auto& x : a.template View<-1u>()) {
    x = dist(rnd);
  }
  for (auto& x : b.template View<-1u>()) {
    x = dist(rnd);
  }
  size_t start = std::clock();
  auto c = MatrixProduct(a, b);
  size_t stop = std::clock();
  auto sm = Sum(c);
  (void)sm;
  return (stop - start) * size_t(1e9) / CLOCKS_PER_SEC;
}

template<class TFloat, size_t N>
size_t AverageNS(size_t iters) {
  size_t sm = 0;
  for (size_t i = 0; i < iters; ++i) {
    sm += SingleRun<TFloat, N>();
  }
  return sm / iters;
}

template<class TFloat, size_t N1, size_t... N>
void Benchmark(size_t iters) {
  std::cout << N1 << ": " << AverageNS<TFloat, N1>(iters) << std::endl;
  if constexpr (sizeof...(N) != 0) {
    Benchmark<TFloat, N...>(iters);
  }
}

int main() {
  std::cout << "Float:" << std::endl;
  Benchmark<float, 5, 10, 16, 25, 32, 50, 64, 100, 128, 200, 256, 500, 512, 1000, 1024, 1500, 2000, 2048>(1);
  std::cout << "Double:" << std::endl;
  Benchmark<double, 5, 10, 16, 25, 32, 50, 64, 100, 128, 200, 256, 500, 512, 1000, 1024, 1500, 2000, 2048>(1);
}
