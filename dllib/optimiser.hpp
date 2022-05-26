#pragma once

#include <dllib/tensor.hpp>
#include <dllib/autograd.hpp>

namespace dllib {

class IArbitraryOptimizerUnit {
  virtual void ZeroGrad() = 0;

  virtual void Step() = 0;

  virtual void Dump(std::ostream&) = 0;

  virtual void Load(std::istream&) = 0;
};

template<CTensor T>
class IOptimizerUnit : public IArbitraryOptimizerUnit {
 public:
  IOptimizerUnit(TVariable<T>& var) : variable(var) {}

  void ZeroGrad() final {
    variable.ZeroGrad();
  }

 protected:
  TVariable<T>& variable;
};

template<CTensor T>
class SGDOptimizerUnit final : public IOptimizerUnit<T> {
 public:
  SGDOptimizerUnit(TVariable<T>& var, typename T::TData lr) : IOptimizerUnit<T>(var), lr_(lr) {
  }

  void Step() {
    variable->value -= variable->grad * lr_;
    variable->ZeroGrad();
  }

 private:
  using IOptimizerUnit<T>::variable;
  const typename T::TData lr_;
};

template<CTensor T>
class MomentumOptimizerUnit final : public IOptimizerUnit<T> {
 public:
  MomentumOptimizerUnit(TVariable<T>& var, typename T::TData lr, typename T::TData alpha)
    : IOptimizerUnit<T>(var),
      lr_(lr),
      alpha_(alpha),
      momentum_(0) {
  }

  void Step() {
    momentum_ *= alpha_;
    momentum_ += variable->grad;
    variable->value -= momentum_ * lr_;
    ZeroGrad();
  }

 private:
  using IOptimizerUnit<T>::variable;
  using IOptimizerUnit<T>::ZeroGrad;

  const typename T::TData lr_;
  const typename T::TData alpha_;
  T momentum_;
};

}  // namespace dllib
