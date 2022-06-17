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

std::mt19937 entropy(std::random_device{}());

template<class TData>
auto GetNormalGenerator() {
  return [
    &rnd = entropy,
    distribution = std::normal_distribution<TData>{}]() mutable {

    return distribution(rnd);
  };
}

template<class TGen, class TDouble>
bool TossCoin(TGen& gen, TDouble p = 0.5) {
  std::uniform_real_distribution<TDouble> dist;
  return dist(gen) < p;
}

}  // namespace helpers

template<class TData, size_t Dim>
struct Bias {
 public:
  Bias() : Bias(helpers::GetNormalGenerator<TData>()) {
  }

  template<class TGen>
  explicit Bias(TGen& gen) {
    for (auto& x : bias->value) {
      x = gen();
    }
  }

  auto operator()(const auto& value) {
    if constexpr (VIsTensor<decltype(value)>) {
      return helpers::AddBias(value, bias->value);
    } else {
      return helpers::AddBias(value, bias);
    }
  }

  auto GetSerializationFields() const {
    return std::tie(bias);
  }

  auto GetParameters() {
    return std::tie(bias);
  }

 private:
  TVariable<TTensor<TData, Dim>> bias{true};
};

template<class TData, size_t From, size_t To>
class FullyConnected {
 public:
  FullyConnected() : FullyConnected(helpers::GetNormalGenerator<TData>()) {}

  template<class TGen>
  explicit FullyConnected(TGen gen) : bias(gen) {
    for (auto& x : var->value.template View<-1u>()) {
      x = gen();
    }
  }

  auto operator()(const auto& value) {
    if constexpr (VIsTensor<decltype(value)>) {
      auto result = MatrixProduct(value, var->value);
      return bias(result);
    } else {
      auto result = MatrixProduct(value, var);
      return bias(result);
    }
  }

  auto GetParameters() {
    return std::tie(var, bias);
  }

  auto GetSerializationFields() const {
    return std::tie(var, bias);
  }

 private:
  TVariable<TTensor<TData, From, To>> var{true};
  Bias<TData, To> bias;
};

template<class TDouble = float>
auto DropOut(const auto& inp, TDouble p = 0.5) {
  using TInput = std::remove_cvref_t<decltype(inp)>;

  if constexpr (VIsTensor<TInput>) {
    return inp;
  } else {
    using T = typename TInput::TUnderlying;
    constexpr size_t batch_size = T::Dimensions[0];
    constexpr size_t channels = T::Dimensions[1];

    auto& gen = helpers::entropy;
    TTensor<bool, batch_size, channels> alive;
    for (auto& x : alive.template View<-1u>()) {
      x = helpers::TossCoin(gen, 1 - p);
    }

    struct TDropOut {
      auto DropOut(const T& val) {
        auto multiply = []<CTensor T>(const T& tensor, bool value) {
          return tensor * value;
        };

        return ApplyFunction</*DimsToSkip=*/2>(multiply, val, alive_);
      }

      auto Forward(const T& val) {
        return DropOut(val) / (1 - p_);
      }

      auto Backward(const T& grad, T* parent) {
        if (parent) {
          *parent += DropOut(grad);
        }
      }

      const TTensor<bool, batch_size, channels> alive_;
      const TDouble p_;
    };

    return TInput(std::make_shared<TOperationNode<TDropOut, T>>(TDropOut{alive, p}, inp));
  }
}

}  // namespace dllib
