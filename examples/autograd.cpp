#include <dllib/tensor.hpp>
#include <dllib/autograd.hpp>

#include <iostream>
#include <random>

void LogExample() {
  dllib::TVariable<dllib::TTensor<float, 2, 2>> var({{1, 2},
                                                     {3, 4}}, true);
  auto lg = dllib::Log(var);
  std::cout << lg->value << std::endl;
  dllib::Sum(lg)->Backward();
  std::cout << var->grad << std::endl;
}

void SqrtExample() {
  dllib::TVariable<dllib::TTensor<float, 2, 2>> var({{.01, 2},
                                                     {  3, 4}}, true);
  auto sqrt = dllib::Sqrt(var);
  std::cout << sqrt->value << std::endl;
  dllib::Sum(sqrt)->Backward();
  std::cout << var->grad << std::endl;
}

size_t FibonacciBenchmark() {
  using TDouble = float;
  using namespace dllib;

  constexpr size_t N = 100, M = 100, K = 10'000;
  std::array<TVariable<TTensor<TDouble, N, N>>, M> arr;

  std::mt19937 rnd(std::random_device{}());
  auto gen = [
    &rnd,
    distribution = std::normal_distribution<TDouble>{}]() mutable {

    return distribution(rnd);
  };

  for (auto& var : arr) {
    for (auto& x: var->value.View<-1u>()) {
      x = gen();
    }
  }

  auto start = clock();
  for (size_t i = 0; i < K; ++i) {
    size_t a = rnd() % M, b = rnd() % M;
    if (rnd() % 2 == 0) {
      arr[a] = arr[a] + arr[b];
    } else {
      arr[a] = arr[b] - arr[a];
    }
  }
  auto sm = arr[0];
  for (size_t i = 1; i < M; ++i) {
    sm = sm + arr[i];
  }
  Sum(sm)->Backward();
  auto stop = clock();

  return (stop - start) * 1000 / CLOCKS_PER_SEC;
}

int main() {
//  LogExample();
//  SqrtExample();
  size_t total_ms = 0;
  for (size_t i = 0; i < 20; ++i) {
    total_ms += FibonacciBenchmark();
  }
  std::cout << total_ms / 20 << std::endl;
}
