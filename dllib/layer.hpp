#pragma once

#include <dllib/tensor.hpp>
#include <dllib/autograd.hpp>

#include <random>

namespace dllib {

namespace helpers {

template<class TData, size_t BatchSize, size_t FirstDim, size_t... OtherDims>
TTensor<TData, BatchSize, FirstDim, OtherDims...> AddBias(
  const TTensor<TData, BatchSize, FirstDim, OtherDims...>& t,
  const TTensor<TData, FirstDim>& bias) {

  auto result = t;
  for (size_t i = 0; i < BatchSize; ++i) {
    for (size_t j = 0; j < FirstDim; ++j) {
      result[i][j] += bias[j];
    }
  }
  return result;
}

template<class TData, size_t BatchSize, size_t FirstDim, size_t... OtherDims>
TVariable<TTensor<TData, BatchSize, FirstDim, OtherDims...>> AddBias(
  const TVariable<TTensor<TData, BatchSize, FirstDim, OtherDims...>>& t,
  const TVariable<TTensor<TData, FirstDim>>& bias) {

  using T = TTensor<TData, BatchSize, FirstDim, OtherDims...>;
  using TBias = TTensor<TData, FirstDim>;

  struct TAddBias {
    auto Forward(const T& t, const TBias& bias) {
      return AddBias(t, bias);
    }

    void Backward(const T& grad, T* t, TBias* bias) {
      if (t) {
        *t += grad;
      }
      if (bias) {
        for (size_t i = 0; i < BatchSize; ++i) {
          for (size_t j = 0; j < FirstDim; ++j) {
            (*bias)[j] += Sum(grad[i][j]);
          }
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
