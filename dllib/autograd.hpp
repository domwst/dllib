#pragma once

#include <dllib/tensor.hpp>

#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>

namespace dllib {

struct IArbitraryVariable;

using TArbitraryVariable = std::shared_ptr<IArbitraryVariable>;

struct IArbitraryVariable {
  // NOLINTNEXTLINE
  IArbitraryVariable(bool requires_grad) : requires_grad(requires_grad) {}

  [[nodiscard]] virtual std::vector<TArbitraryVariable> GetChildren() const = 0;

  virtual void PushGradient() = 0;

  virtual ~IArbitraryVariable() = default;

  bool requires_grad;
};

template<CTensor T>
struct IVariable;

template<CTensor T>
class TVariable : public std::shared_ptr<IVariable<T>> {
 public:
  using std::shared_ptr<IVariable<T>>::shared_ptr;

  // NOLINTNEXTLINE
  TVariable(T value) : TVariable(std::make_shared<IVariable<T>>(value)) {
  }
};


template<CTensor T>
struct IVariable : public IArbitraryVariable {
  using IArbitraryVariable::requires_grad;

  // NOLINTNEXTLINE
  IVariable(const T& value, bool requires_grad = false) : IArbitraryVariable(requires_grad), value(value) {}

  ~IVariable() override = default;

  template<class U = T>
  std::enable_if_t<U::DimensionCount == 0, void> Backward() {
    std::unordered_set<TArbitraryVariable> SubGraph;
    std::vector<TArbitraryVariable> order;

    auto dfs = [&SubGraph, &order](auto& self, TArbitraryVariable v) -> void {
      SubGraph.insert(v);
      for (auto&& child: v->GetChildren()) {
        if (child->requires_grad && !SubGraph.count(child)) {
          self(self, std::move(child));
        }
      }
      order.emplace_back(std::move(v));
    };

    for (auto&& child: GetChildren()) {
      if (child->requires_grad && !SubGraph.contains(child)) {
        dfs(dfs, std::move(child));
      }
    }

    grad = 1;
    PushGradient();
    while (!order.empty()) {
      order.back()->PushGradient();
      order.pop_back();
    }
  }

  void zero_grad() {
    grad.FillWith(0);
  }

  T value;
  T grad;
};


template<CTensor T>
struct TLeafNode final : public IVariable<T> {
  using IVariable<T>::IVariable;
  using IVariable<T>::value;
  using IVariable<T>::grad;
  using IVariable<T>::requires_grad;
  using IVariable<T>::zero_grad;

  [[nodiscard]] std::vector<TArbitraryVariable> GetChildren() const {
    return {};
  }

  void PushGradient() {}
};

template<CTensor T>
inline constexpr TVariable<T> MakeLeaf(const T& value, bool requires_grad = false) {
  return std::make_shared<TLeafNode<T>>(value, requires_grad);
}


namespace helpers {

template<class TFirstArg, class... TArgs>
constexpr bool CalculateOr(TFirstArg first_arg, TArgs... other_args) {
  if constexpr (sizeof...(TArgs) == 0) {
    return first_arg;
  } else {
    return first_arg || CalculateOr(other_args...);
  }
}

template<CTensor T>
constexpr std::optional<T*> GetGradientPointerIfRequired(const TVariable<T>& v) {
  if (v->requires_grad) {
    return &(v->grad);
  } else {
    return {};
  }
}

}  // namespace helpers


template<class TOperation, CTensor... TArgs>
struct TOperationNode : public IVariable<std::invoke_result_t<decltype(&TOperation::Forward), TOperation*, TArgs...>> {
  using TValue = std::invoke_result_t<decltype(&TOperation::Forward), TOperation*, TArgs...>;

  using IVariable<TValue>::IVariable;
  using IVariable<TValue>::value;
  using IVariable<TValue>::grad;
  using IVariable<TValue>::requires_grad;
  using IVariable<TValue>::zero_grad;

  // NOLINTNEXTLINE
  TOperationNode(TOperation op, const TVariable<TArgs>& ... args) :
    IVariable<TValue>(op.Forward(args->value...), helpers::CalculateOr(args->requires_grad...)),
    operation_(std::move(op)),
    args_({args...}) {}

  TOperationNode(TOperationNode&&) noexcept = default;

  [[nodiscard]] std::vector<TArbitraryVariable> GetChildren() const {
    return [this]<size_t... i>(std::index_sequence<i...>) -> std::vector<TArbitraryVariable>{
      return { static_pointer_cast<IArbitraryVariable>(get<i>(args_))... };
    }(std::make_index_sequence<sizeof...(TArgs)>());
  }

  void PushGradient() {
    [this]<size_t... i>(std::index_sequence<i...>) {
      operation_.Backward(grad, helpers::GetGradientPointerIfRequired(get<i>(args_))...);
    }
    (std::make_index_sequence<sizeof...(TArgs)>());
    zero_grad();
  }

 private:
  TOperation operation_;
  std::tuple<TVariable<TArgs>...> args_;
};

template<CTensor T>
TVariable<T> operator+(const TVariable<T>& l, const TVariable<T>& r) {
  struct TAddition {
    T Forward(const T& l, const T& r) {
      return l + r;
    }

    void Backward(const T& grad, std::optional<T*> l, std::optional<T*> r) {
      if (l) {
        **l += grad;
      }
      if (r) {
        **r += grad;
      }
    }
  };

  return std::make_shared<TOperationNode<TAddition, T, T>>(TAddition{}, l, r);
}

template<CTensor T>
TVariable<T> operator-(const TVariable<T>& l, const TVariable<T>& r) {
  struct TSubtraction {
    T Forward(const T& l, const T& r) {
      return l - r;
    }

    void Backward(const T& grad, std::optional<T*> l, std::optional<T*> r) {
      if (l) {
        **l += grad;
      }
      if (r) {
        **r -= grad;
      }
    }
  };

  return std::make_shared<TOperationNode<TSubtraction, T, T>>(TSubtraction{}, l, r);
}

template<CTensor T>
TVariable<T> operator*(const TVariable<T>& l, const TVariable<T>& r) {
  struct TMultiplication {
    T Forward(const T& l, const T& r) {
      l_ = l;
      r_ = r;
      return l * r;
    }

    void Backward(const T& grad, std::optional<T*> l, std::optional<T*> r) {
      if (l) {
        **l += grad * r_;
      }
      if (r) {
        **r = l_ * grad;
      }
    }

    T l_, r_;
  };

  return std::make_shared<TOperationNode<TMultiplication, T, T>>(TMultiplication{}, l, r);
}

template<CTensor T1, CTensor T2>
auto MatrixMultiplication(const IVariable<T1>& l, const IVariable<T2>& r) {
  struct TMatrixMultiplication {
    helpers::MatrixMultiplicationResult<T1, T2> Forward(const T1& l, const T2& r) {
      l_ = l;
      r_ = r;
      return MatrixMultiplication(l, r);
    }

    void Backward(const auto& grad, std::optional<T1*> l, std::optional<T2*> r) {
      if (l) {
        MatrixMultiplication(grad, r_.T(), **l);
      }
      if (r) {
        MatrixMultiplication(l_.T(), grad, **r);
      }
    };

    T1 l_;
    T2 r_;
  };

  return std::make_shared<TOperationNode<TMatrixMultiplication, T1, T2>>(TMatrixMultiplication{}, l, r);
}

// template<CTensor T>
// struct TSumAllNode : public IVariable<TTensor<typename T::DataType>> {
//     using ValueType = TTensor<typename T::DataType>;

//     using IVariable<ValueType>::value;
//     using IVariable<ValueType>::grad;
//     using IVariable<ValueType>::requires_grad;
//     using IVariable<ValueType>::zero_grad;

//     TSumAllNode(const TVariable<T>& arg) :
//         IVariable<ValueType>(Sum(arg->value)),
//         v_(arg) {}

//     void PushGradient() {
//         v_->grad += grad;
//         zero_grad();
//     }

//     std::vector<TArbitraryVariable> GetChildren() const {
//         return {v_};
//     }

// private:
//     TVariable<T> v_;
// };

// template<CTensor T>
// TVariable<TTensor<typename T::DataType>> Sum(const TVariable<T>& arg) {
//     return std::make_shared<TSumAllNode<T>>(arg);
// }

// template<CTensor T>
// struct TAddScalarNode : public IVariable<T> {
//     using IVariable<T>::value;
//     using IVariable<T>::grad;
//     using IVariable<T>::requires_grad;
//     using IVariable<T>::zero_grad;

//     TAddScalarNode(const TVariable<T>& l, const TVariable<typename T::DataType>& r) :
//         IVariable<T>(l->value() + r->value()),
//         l_(l), r_(r) {}

//     TAddScalarNode(const TVariable<typename T::DataType>& l, const TVariable<T>& r) :
//         TAddScalarNode(r, l) {}

//     void PushGradient() {
//         l_ += grad;
//         r_ += Sum(grad);
//     }

//     std::vector<TArbitraryVariable> GetChildren() const {
//         return {l_, r_};
//     }

// private:
//     TVariable<T> l_;
//     TVariable<typename T::DataType> r_;
// };

}