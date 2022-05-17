#pragma once

#include <dllib/tensor.hpp>

#include <memory>
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
struct TLeafNode;

template<class TOperation, CTensor... TArgs>
struct TOperationNode;

template<CTensor TT>
struct TVariable : public std::shared_ptr<IVariable<TT>> {
  using std::shared_ptr<IVariable<TT>>::shared_ptr;

  // NOLINTNEXTLINE
  TVariable(const TT& value, bool required_grad = false)
    : std::shared_ptr<IVariable<TT>>(std::make_shared<TLeafNode<TT>>(value, required_grad)) {
  }

  template<size_t... NewDims>
  TVariable<TTensor<typename TT::DataType, NewDims...>> View() const {
    struct TView {
      _Pragma("clang diagnostic ignored \"-Wunused-local-typedef\"")
      using ConvertedTensor = TTensor<typename TT::DataType, NewDims...>;
      _Pragma("clang diagnostic warning \"-Wunused-local-typedef\"")

      ConvertedTensor Forward(const TT& val) {
        return val.template View<NewDims...>();
      }

      void Backward(const ConvertedTensor& grad, TT* v) {
        if (v) {
          [v, &grad]<size_t... i>(std::index_sequence<i...>) {
            *v += grad.template View<TT::Dimensions[i]...>();
          }(std::make_index_sequence<TT::DimensionCount>{});
        }
      }
    };

    return std::make_shared<TOperationNode<TView, TT>>(TView{}, *this);
  }

  template<class U = TT, class TTransposeResult = helpers::TTransposeResult<U>>
  TVariable<helpers::TTransposeResult<U>> T() const {
    struct TTranspose {
      TTransposeResult Forward(const U& val) {
        return val.T();
      }

      void Backward(const TTransposeResult& grad, U* v) {
        if (v) {
          *v += grad.T();
        }
      }
    };

    return std::make_shared<TOperationNode<TTranspose, TT>>(TTranspose{}, *this);
  }
};


template<CTensor T>
struct IVariable : public IArbitraryVariable {
  using IArbitraryVariable::requires_grad;

  // NOLINTNEXTLINE
  IVariable(const T& value, bool requires_grad = false) : IArbitraryVariable(requires_grad), value(value), grad(0) {}

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

  void ZeroGrad() {
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
  using IVariable<T>::ZeroGrad;

  [[nodiscard]] std::vector<TArbitraryVariable> GetChildren() const {
    return {};
  }

  void PushGradient() {}
};

namespace helpers {

constexpr bool CalculateOr(auto... args) {
  return (args || ... || false);
}

template<CTensor T>
constexpr T* GetGradientPointerIfRequired(const TVariable<T>& v) {
  if (v->requires_grad) {
    return &(v->grad);
  } else {
    return nullptr;
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
  using IVariable<TValue>::ZeroGrad;

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
    }(std::make_index_sequence<sizeof...(TArgs)>());
    ZeroGrad();
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

    void Backward(const T& grad, T* l, T* r) {
      if (l) {
        *l += grad;
      }
      if (r) {
        *r += grad;
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

    void Backward(const T& grad, T* l, T* r) {
      if (l) {
        *l += grad;
      }
      if (r) {
        *r -= grad;
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

    void Backward(const T& grad, T* l, T* r) {
      if (l) {
        *l += grad * r_;
      }
      if (r) {
        *r += l_ * grad;
      }
    }

    T l_, r_;
  };

  return std::make_shared<TOperationNode<TMultiplication, T, T>>(TMultiplication{}, l, r);
}

template<CTensor T1, CTensor T2>
TVariable<helpers::TMatrixProductResult<T1, T2>> MatrixProduct(const TVariable<T1>& l, const TVariable<T2>& r) {
  struct TMatrixProduct {
    helpers::TMatrixProductResult<T1, T2> Forward(const T1& l, const T2& r) {
      l_ = l;
      r_ = r;
      return MatrixProduct(l, r);
    }

    void Backward(const helpers::TMatrixProductResult<T1, T2>& grad, T1* l, T2* r) {
      if (l) {
        MatrixProduct(grad, r_.T(), *l);
      }
      if (r) {
        MatrixProduct(l_.T(), grad, *r);
      }
    };

    T1 l_;
    T2 r_;
  };

  return std::make_shared<TOperationNode<TMatrixProduct, T1, T2>>(TMatrixProduct{}, l, r);
}

template<CTensor T>
auto Sum(const TVariable<T>& val) {
  struct TSum {
    TTensor<typename T::DataType> Forward(const T& val) {
      return Sum(val);
    }

    void Backward(typename T::DataType grad, T* v) {
      if (v) {
        *v += grad;
      }
    }
  };

  return std::make_shared<TOperationNode<TSum, T>>(TSum{}, val);
}

}
