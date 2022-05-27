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
class SGDOptimizerUnit final : public IOptimizerUnit<T> {
 public:
  SGDOptimizerUnit(TVariable<T>& var, typename T::TData lr) : IOptimizerUnit<T>(var), lr_(lr) {
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
class MomentumOptimizerUnit final : public IOptimizerUnit<T> {
 private:
  using TData = typename T::TData;

 public:
  MomentumOptimizerUnit(TVariable<T>& var, TData lr, TData alpha)
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
class AdamOptimizerUnit final : public IOptimizerUnit<T> {
 private:
  using TData = typename T::TData;

 public:
  AdamOptimizerUnit(
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

}  // namespace dllib
