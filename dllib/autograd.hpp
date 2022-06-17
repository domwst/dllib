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
 private:
  template<size_t... NewDims>
  using ViewResult = std::remove_reference_t<decltype(std::declval<TT>().template View<NewDims...>())>;

 public:
  using TUnderlying = TT;
  using std::shared_ptr<IVariable<TT>>::shared_ptr;

  TVariable() : TVariable(false) {
  }

  explicit TVariable(bool requires_grad) : TVariable(TT{}, requires_grad) {
  }

  TVariable(const TT& value, bool required_grad)
    : std::shared_ptr<IVariable<TT>>(std::make_shared<TLeafNode<TT>>(value, required_grad)) {
  }

  [[nodiscard]] bool IsLeaf() const {
    auto ptr = dynamic_cast<TLeafNode<TT>*>(std::shared_ptr<IVariable<TT>>::get());
    return ptr != nullptr;
  }

  [[nodiscard]] TVariable Copy() const {
    return TVariable((*this)->value, (*this)->requires_grad);
  }

  template<size_t... NewDims>
  [[nodiscard]] TVariable<ViewResult<NewDims...>> View() const {
    struct TView {
      _Pragma("clang diagnostic ignored \"-Wunused-local-typedef\"")
      using ConvertedTensor = ViewResult<NewDims...>;
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

  TVariable operator-() const {
    struct TNeg {
      TT Forward(const TT& val) const {
        return -val;
      }

      void Backward(const TT& grad, TT* v) {
        if (v) {
          *v -= grad;
        }
      }
    };

    return std::make_shared<TOperationNode<TNeg, TT>>(TNeg{}, *this);
  }

  std::tuple<TT&, TT&> GetSerializationFields() const {
    return {(*this)->value, (*this)->grad};
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

  using IVariable<TValue>::value;
  using IVariable<TValue>::grad;
  using IVariable<TValue>::requires_grad;
  using IVariable<TValue>::ZeroGrad;

  // NOLINTNEXTLINE
  TOperationNode(TOperation op, const TVariable<TArgs>& ... args) :
    IVariable<TValue>(op.Forward(args->value...), helpers::CalculateOr(args->requires_grad...)),
    operation_(std::move(op)),
    args_({args...}) {

    if (!requires_grad) {
      args_ = {};
    }
  }

  TOperationNode(TOperationNode&&) noexcept = default;

  [[nodiscard]] std::vector<TArbitraryVariable> GetChildren() const {
    return [this]<size_t... i>(std::index_sequence<i...>) -> std::vector<TArbitraryVariable>{
      return { static_pointer_cast<IArbitraryVariable>(get<i>(args_))... };
    }(std::make_index_sequence<sizeof...(TArgs)>());
  }

  void PushGradient() {
    [this]<size_t... i>(std::index_sequence<i...>) {
      constexpr bool pointers_callable = std::is_invocable_v<
        decltype(&TOperation::Backward),
        TOperation*,
        const TValue&,
        decltype(&(get<i>(args_)->grad))...>;

      constexpr bool variables_callable = std::is_invocable_v<
        decltype(&TOperation::Backward),
        TOperation*,
        const TValue&,
        decltype(get<i>(args_))...>;

      constexpr bool callable_with_current_variable = std::is_invocable_v<
        decltype(&TOperation::Backward),
        TOperation*,
        const IVariable<TValue>*,
        decltype(get<i>(args_))...>;

      static_assert(pointers_callable || variables_callable || callable_with_current_variable,
        "TOperation::Backward should be callable either with " \
        "pointers to tensors or with variables as arguments");

      if constexpr (pointers_callable) {
        operation_.Backward(grad, helpers::GetGradientPointerIfRequired(get<i>(args_))...);
      } else if constexpr (variables_callable) {
        operation_.Backward(grad, get<i>(args_)...);
      } else if constexpr (callable_with_current_variable) {
        operation_.Backward(this, get<i>(args_)...);
      }
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
      return l * r;
    }

    void Backward(const T& grad, TVariable<T>& l, TVariable<T>& r) {
      if (l->requires_grad) {
        l->grad += grad * r->value;
      }
      if (r->requires_grad) {
        r->grad += l->value * grad;
      }
    }
  };

  return std::make_shared<TOperationNode<TMultiplication, T, T>>(TMultiplication{}, l, r);
}

template<CTensor T1, CTensor T2>
TVariable<helpers::TMatrixProductResult<T1, T2>> MatrixProduct(const TVariable<T1>& l, const TVariable<T2>& r) {
  struct TMatrixProduct {
    helpers::TMatrixProductResult<T1, T2> Forward(const T1& l, const T2& r) {
      return MatrixProduct(l, r);
    }

    void Backward(const helpers::TMatrixProductResult<T1, T2>& grad, TVariable<T1>& l, TVariable<T2>& r) {
      if (l->requires_grad) {
        MatrixProduct(grad, r->value.T(), l->grad);
      }
      if (r->requires_grad) {
        MatrixProduct(l->value.T(), grad, r->grad);
      }
    };
  };

  return std::make_shared<TOperationNode<TMatrixProduct, T1, T2>>(TMatrixProduct{}, l, r);
}

template<CTensor T>
TVariable<T> Log(const TVariable<T>& val) {
  struct TLog {
    T Forward(const T& val) {
      return Log(val);
    }

    void Backward(const T& grad, TVariable<T>& parent) {
      if (parent->requires_grad) {
        parent->grad += grad / parent->value;
      }
    }
  };

  return std::make_shared<TOperationNode<TLog, T>>(TLog{}, val);
}

template<CTensor T>
TVariable<T> Sqrt(const TVariable<T>& val) {
  struct TSqrt {
    T Forward(const T& val) {
      return Sqrt(val);
    }

    void Backward(const IVariable<T>* current, TVariable<T>& parent) {
      if (parent->requires_grad) {
        parent->grad += T(1. / 2) / current->value;
      }
    }
  };

  return std::make_shared<TOperationNode<TSqrt, T>>(TSqrt{}, val);
}

template<size_t Dim, CTensor T1, CTensor T2>
TVariable<helpers::TStackAlongResult<Dim, T1, T2>> StackAlong(
  const TVariable<T1>& v1,
  const TVariable<T2>& v2) {

  using TStackAlongResult = helpers::TStackAlongResult<Dim, T1, T2>;

  struct TStackAlong {
    TStackAlongResult Forward(const T1& l, const T2& r) {
      return StackAlong<Dim>(l, r);
    }

    void Backward(const TStackAlongResult& grad, T1* l, T2* r) {
      auto [l_grad, r_grad] = SplitAlong<Dim, T1::Dimensions[Dim]>(grad);
      if (l) {
        *l += l_grad;
      }
      if (r) {
        *r += r_grad;
      }
    }
  };

  return std::make_shared<TOperationNode<TStackAlong, T1, T2>>(TStackAlong{}, v1, v2);
}

template<CTensor T>
auto Sum(const TVariable<T>& val) {
  struct TSum {
    TTensor<typename T::TData> Forward(const T& val) {
      return Sum(val);
    }

    void Backward(typename T::TData grad, T* v) {
      if (v) {
        *v += grad;
      }
    }
  };

  return std::make_shared<TOperationNode<TSum, T>>(TSum{}, val);
}

template<CTensor T>
TVariable<T> Exp(const TVariable<T>& val) {
  struct TExp {
    T Forward(const T& val) {
      return Exp(val);
    }

    void Backward(const IVariable<T>* current, TVariable<T>& parent) {
      if (parent->requires_grad) {
        parent->grad += current->grad * current->value;
      }
    }
  };

  return std::make_shared<TOperationNode<TExp, T>>(TExp{}, val);
}

template<CTensor T>
TVariable<T> Tanh(const TVariable<T>& val) {
  struct TTanh {
    T Forward(const T& val) {
      return Tanh(val);
    }

    void Backward(const IVariable<T>* current, TVariable<T>& parent) {
      if (parent->requires_grad) {
        parent->grad += current->grad * (1 - current->value * current->value);
      }
    }
  };

  return std::make_shared<TOperationNode<TTanh, T>>(TTanh{}, val);
}

template<CTensor T>
TVariable<T> Sigmoid(const TVariable<T>& val) {
  struct TSigmoid {
    T Forward(const T& val) {
      return Sigmoid(val);
    }

    void Backward(const IVariable<T>* current, TVariable<T>& parent) {
      if (parent->requires_grad) {
        parent->grad += current->grad * current->value * (1 - current->value);
      }
    }
  };

  return std::make_shared<TOperationNode<TSigmoid, T>>(TSigmoid{}, val);
}

}
