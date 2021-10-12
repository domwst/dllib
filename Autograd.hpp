#pragma once

#include "Tensor.hpp"

#include <memory>
#include <vector>
#include <unordered_set>

namespace dllib {

struct IArbitraryVariable;

using TArbitraryVariable = std::shared_ptr<IArbitraryVariable>;

struct IArbitraryVariable {
    IArbitraryVariable(bool requires_grad) : requires_grad(requires_grad) {}

    virtual std::vector<TArbitraryVariable> GetChildren() const = 0;
    virtual void PushGradient() = 0;
    virtual ~IArbitraryVariable() = default;

    bool requires_grad;
};

template<CTensor T>
struct IVariable;

template<CTensor T>
using TVariable = std::shared_ptr<IVariable<T>>;


template<CTensor T>
struct IVariable : public IArbitraryVariable {
    using IArbitraryVariable::requires_grad;

    IVariable(const T& value, bool requires_grad = false) : IArbitraryVariable(requires_grad), value(value) {}

    virtual ~IVariable() = default;

    std::enable_if_t<T::DimensionCount == 0, void> backward() {
        std::unordered_set<TArbitraryVariable> SubGraph;
        std::vector<TArbitraryVariable> order;

        auto dfs = [&SubGraph, &order](auto self, TArbitraryVariable v) -> void {
            SubGraph.insert(v);
            for (auto&& child : v->GetChildren()) {
                if (child->requires_grad && !SubGraph.count(child)) {
                    self(self, std::move(child));
                }
            }
            order.emplace_back(std::move(v));
        };

        for (auto&& child : GetChildren()) {
            if (child->requires_grad && !SubGraph.count(child)) {
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

    std::vector<TArbitraryVariable> GetChildren() const {
        return {};
    }

    void PushGradient() {}
};

template<CTensor T>
inline constexpr TVariable<T> MakeLeaf(const T& value, bool requires_grad = false) {
    return std::make_shared<TLeafNode<T>>(value, requires_grad);
}


template<CTensor T>
struct TAddNode final : public IVariable<T> {
    using IVariable<T>::value;
    using IVariable<T>::grad;
    using IVariable<T>::requires_grad;
    using IVariable<T>::zero_grad;

    TAddNode(const TVariable<T>& l, const TVariable<T>& r) :
        IVariable<T>(l->value + r->value, l->requires_grad || r->requires_grad),
        l_(l), r_(r) {}

    void PushGradient() {
        if (l_->requires_grad) {
            l_->grad += grad;
        }
        if (r_->requires_grad) {
            r_->grad += grad;
        }
        zero_grad();
    }

    std::vector<TArbitraryVariable> GetChildren() const {
        return {l_, r_};
    }

private:
    TVariable<T> l_;
    TVariable<T> r_;
};

template<CTensor T>
TVariable<T> operator+(const TVariable<T>& l, const TVariable<T>& r) {
    return std::make_shared<TAddNode<T>>(l, r);
}


template<CTensor T>
struct TSubtractNode final : public IVariable<T> {
    using IVariable<T>::value;
    using IVariable<T>::grad;
    using IVariable<T>::requires_grad;
    using IVariable<T>::zero_grad;

    TSubtractNode(const TVariable<T>& l, const TVariable<T>& r) :
        IVariable<T>(l->value - r->value, l->requires_grad || r->requires_grad),
        l_(l), r_(r) {}

    void PushGradient() {
        if (l_->requires_grad) {
            l_->grad += grad;
        }
        if (r_->requires_grad) {
            r_->grad -= grad;
        }
        zero_grad();
    }

    std::vector<TArbitraryVariable> GetChildren() const {
        return {l_, r_};
    }

private:
    TVariable<T> l_;
    TVariable<T> r_;
};

template<CTensor T>
TVariable<T> operator-(const TVariable<T>& l, const TVariable<T>& r) {
    return std::make_shared<TSubtractNode<T>>(l, r);
}


template<CTensor T>
struct TMultiplyNode final : public IVariable<T> {
    using IVariable<T>::value;
    using IVariable<T>::grad;
    using IVariable<T>::requires_grad;
    using IVariable<T>::zero_grad;

    TMultiplyNode(const TVariable<T>& l, const TVariable<T>& r) :
        IVariable<T>(l->value * r->value, l->requires_grad || r->requires_grad),
        l_(l), r_(r) {}

    void PushGradient() {
        if (l_->requires_grad) {
            l_->grad += grad * r_->value;
        }
        if (r_->requires_grad) {
            r_->grad += grad * l_->value;
        }
        zero_grad();
    }

    std::vector<TArbitraryVariable> GetChildren() const {
        return {l_, r_};
    }

private:
    TVariable<T> l_;
    TVariable<T> r_;
};

template<CTensor T>
TVariable<T> operator*(const TVariable<T>& l, const TVariable<T>& r) {
    return std::make_shared<TMultiplyNode<T>>(l, r);
}

}