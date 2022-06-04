#pragma once

#include <dllib/tensor.hpp>
#include <dllib/autograd.hpp>

#include <random>

namespace dllib {

namespace helpers {

template<class TData, size_t BatchSize, size_t... OtherDims>
TTensor<TData, BatchSize, OtherDims...> AddBias(
  const TTensor<TData, BatchSize, OtherDims...>& t,
  const TTensor<TData, OtherDims...>& bias) {

  auto result = t;
  for (auto& x : result) {
    x += bias;
  }
  return result;
}

template<class TData, size_t BatchSize, size_t... OtherDims>
TVariable<TTensor<TData, BatchSize, OtherDims...>> AddBias(
  const TVariable<TTensor<TData, BatchSize, OtherDims...>>& t,
  const TVariable<TTensor<TData, OtherDims...>>& bias) {

  using T = TTensor<TData, BatchSize, OtherDims...>;
  using TBias = TTensor<TData, OtherDims...>;

  struct TAddBias {
    auto Forward(const T& t, const TBias& bias) {
      return AddBias(t, bias);
    }

    void Backward(const T& grad, T* t, TBias* bias) {
      if (t) {
        *t += grad;
      }
      if (bias) {
        for (auto& x : grad) {
          *bias += x;
        }
      }
    }
  };

  return std::make_shared<TOperationNode<TAddBias, T, TBias>>(TAddBias{}, t, bias);
}

}  // namespace helpers

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
    for (auto& x : var->value.template View<-1u>()) {
      x = gen();
    }
    for (auto& x : bias->value.template View<-1u>()) {
      x = gen();
    }
  }

  auto operator()(const auto& value) {
    if constexpr (VIsTensor<decltype(value)>) {
      auto result = MatrixProduct(value, var->value);
      return helpers::AddBias(result, bias->value);
    } else {
      auto result = MatrixProduct(value, var);
      return helpers::AddBias(result, bias);
    }
  }

  std::tuple<TVariable<TTensor<TData, From, To>>&, TVariable<TTensor<TData, To>>&> GetParameters() {
    return {var, bias};
  }

  std::tuple<const TVariable<TTensor<TData, From, To>>&, const TVariable<TTensor<TData, To>>&>
  GetSerializationFields() const {
    return {var, bias};
  }

 private:
  TVariable<TTensor<TData, From, To>> var{true};
  TVariable<TTensor<TData, To>> bias{true};
};

}  // namespace dllib
