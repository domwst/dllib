#pragma once

#include <dllib/tensor.hpp>
#include <dllib/autograd.hpp>

#include <random>

namespace dllib {

template<class TData, size_t From, size_t To>
class FullyConnected {
 public:
  FullyConnected() : FullyConnected([
    rnd = std::mt19937(std::random_device{}()),
    distribution = std::normal_distribution<TData>{}]() mutable {

    return distribution(rnd);
  }) {}

  template<class TGen>
  explicit FullyConnected(TGen gen) {
    auto& data = var->value.template View<From * To>();
    std::generate(data.Begin(), data.End(), std::move(gen));
  }

  auto operator()(const auto& value) {
    if constexpr (VIsTensor<decltype(value)>) {
      return MatrixProduct(value, var->value);
    } else {
      return MatrixProduct(value, var);
    }
  }

  std::tuple<TVariable<TTensor<TData, From, To>>&> GetParameters() {
    return {var};
  }

  std::tuple<const TVariable<TTensor<TData, From, To>>&> GetSerializationFields() const {
    return {var};
  }

 private:
  TVariable<TTensor<TData, From, To>> var{true};
};

}  // namespace dllib
