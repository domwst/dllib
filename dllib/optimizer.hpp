#pragma once

#include <dllib/tensor.hpp>
#include <dllib/autograd.hpp>
#include <dllib/serialization.hpp>

namespace dllib {

class IArbitraryOptimizerUnit {
 public:
  virtual void ZeroGrad() = 0;

  void Step() {
    StepImpl();
    ZeroGrad();
  }

  virtual void Dump(std::ostream&) const = 0;

  virtual void Load(std::istream&) = 0;

  virtual ~IArbitraryOptimizerUnit() {}

 protected:
  virtual void StepImpl() = 0;
};

template<CTensor T>
class IOptimizerUnit : public IArbitraryOptimizerUnit {
 public:
  IOptimizerUnit(TVariable<T>& var) : variable(var) {}

  void ZeroGrad() final {
    variable->ZeroGrad();
  }

 protected:
  TVariable<T>& variable;
};

template<CTensor T>
class TSGDOptimizerUnit final : public IOptimizerUnit<T> {
 public:
  TSGDOptimizerUnit(TVariable<T>& var, typename T::TData lr) : IOptimizerUnit<T>(var), lr_(lr) {
  }

  void Dump(std::ostream&) const final {
  }

  void Load(std::istream&) final {
  }

 protected:
  void StepImpl() final {
    variable->value -= variable->grad * lr_;
  }

  using IOptimizerUnit<T>::variable;
  const typename T::TData lr_;
};

template<CTensor T>
class TMomentumOptimizerUnit final : public IOptimizerUnit<T> {
 private:
  using TData = typename T::TData;

 public:
  TMomentumOptimizerUnit(TVariable<T>& var, TData lr, TData alpha)
    : IOptimizerUnit<T>(var),
      lr_(lr),
      alpha_(alpha),
      momentum_(0) {
  }

  void Dump(std::ostream& out) const final {
    dllib::Dump(out, momentum_);
  }

  void Load(std::istream& in) final {
    dllib::Load(in, momentum_);
  }

 protected:
  void StepImpl() final {
    momentum_ *= alpha_;
    momentum_ += variable->grad;
    variable->value -= momentum_ * lr_;
  }

  using IOptimizerUnit<T>::variable;
  using IOptimizerUnit<T>::ZeroGrad;

  const TData lr_;
  const TData alpha_;
  T momentum_;
};

template<CTensor T>
class TAdamOptimizerUnit final : public IOptimizerUnit<T> {
 private:
  using TData = typename T::TData;

 public:
  TAdamOptimizerUnit(
    TVariable<T>& var,
    TData lr,
    TData beta1 = 0.9,
    TData beta2 = 0.999,
    TData eps = 1e-8)
    : IOptimizerUnit<T>(var),
      lr_(lr),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      m_(0),
      v_(0) {
  }

  void Dump(std::ostream& out) const final {
    dllib::Dump(out, beta1_power_);
    dllib::Dump(out, beta2_power_);
    dllib::Dump(out, m_);
    dllib::Dump(out, v_);
  }

  void Load(std::istream& in) final {
    dllib::Load(in, beta1_power_);
    dllib::Load(in, beta2_power_);
    dllib::Load(in, m_);
    dllib::Load(in, v_);
  }

 protected:
  void StepImpl() final {
    auto& grad = variable->grad;

    m_ *= beta1_;
    m_ += grad * (1 - beta1_);
    beta1_power_ *= beta1_;

    v_ *= beta2_;
    v_ += grad * grad * (1 - beta2_);
    beta2_power_ *= beta2_;

    auto m_hat = m_ / (1 - beta1_power_);
    auto v_hat = v_ / (1 - beta2_power_);

    variable->value -= m_hat / (Sqrt(v_hat) + eps_) * lr_;
  }

  using IOptimizerUnit<T>::variable;
  using IOptimizerUnit<T>::ZeroGrad;

  const TData lr_;

  const typename T::TData beta1_;
  TData beta1_power_ = 1;

  const TData beta2_;
  TData beta2_power_ = 1;

  const TData eps_;
  T m_, v_;
};

template<template<CTensor T> class TOptimizer, class... TParams>
class TOptimizerManager {
 public:
  explicit TOptimizerManager(TParams&&... params)
    : constructor_parameters_(std::forward<TParams>(params)...) {
  }

  template<CTensor T>
  void AddParameter(TVariable<T>& var) {
    [this, &var]<size_t... i>(std::index_sequence<i...>) {
      EmplaceOptimizer(var, get<i>(constructor_parameters_)...);
    }(std::make_index_sequence<sizeof...(TParams)>());
  }

  template<CTensor... TArgs>
  void AddParameters(const std::tuple<TVariable<TArgs>&...>& params) {
    [this, &params]<size_t... i>(std::index_sequence<i...>) {
      //  Just because compiler thinks "this" is unused
      (this->AddParameter(get<i>(params)),...);
    }(std::make_index_sequence<sizeof...(TArgs)>());
  }

  template<CTensor T, class... TArgs>
  void EmplaceOptimizer(TVariable<T>& var, TArgs&&... args) {
    optimizers_.emplace_back(std::make_unique<TOptimizer<T>>(var, std::forward<TArgs>(args)...));
  }

  void Dump(std::ostream& out) const {
    for (auto& opt : optimizers_) {
      opt->Dump(out);
    }
  }

  void Load(std::istream& in) {
    for (auto& opt : optimizers_) {
      opt->Load(in);
    }
  }

  void ZeroGrad() {
    for (auto& opt : optimizers_) {
      opt->ZeroGrad();
    }
  }

  void Step() {
    for (auto& opt : optimizers_) {
      opt->Step();
    }
  }

 private:
  std::vector<std::unique_ptr<IArbitraryOptimizerUnit>> optimizers_;
  std::tuple<TParams...> constructor_parameters_;
};

//  This is necessary because C++ can't deduce only first argument of class
//  template -- either none or all
template<template<CTensor T> class Optimizer, class... TParams>
[[nodiscard]] auto MakeOptimizerManager(TParams&&... params) {
  return TOptimizerManager<Optimizer, TParams...>(std::forward<TParams>(params)...);
}

}  // namespace dllib
