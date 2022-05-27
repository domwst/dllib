#include <boost/ut.hpp>
#include <dllib/optimizer.hpp>
#include <memory>

namespace ut = boost::ut;

template<size_t... Dims>
using Tensor = dllib::TTensor<float, Dims...>;

static ut::suite optimizer_tests = [] {
  using namespace ut;
  using namespace dllib;

  "SGD_simple_1"_test = [] {
    Tensor<2, 3> start = {
      {1, 2, 3},
      {4, 5, 6},
    };
    TVariable v(start, true);
    SGDOptimizerUnit opt(v, 1);

    Sum(v * v)->Backward();
    opt.Step();
    expect(AllClose(v->value, -start));

    Sum(v * v)->Backward();
    opt.Step();
    expect(AllClose(v->value, start));
  };

  "SGD_simple_2"_test = [] {
    Tensor<2, 3> start = {
      {1, 2, 3},
      {4, 5, 6},
    };
    TVariable v(start, true);
    SGDOptimizerUnit opt(v, 1);

    Sum(Log(v))->Backward();
    opt.Step();
    expect(AllClose(v->value, Tensor<2, 3>({
      {0,    1.5, 2.6666666},
      {3.75, 4.8, 5.8333333},
    })));
    expect(AllClose(v->grad, Tensor<2, 3>(0)));
  };

  "SGD_lr_usage"_test = [] {
    Tensor<2, 3> start = {
      {1, 2, 3},
      {4, 5, 6},
    };
    TVariable v(start, true);
    SGDOptimizerUnit opt(v, .5);

    Sum(v * v)->Backward();
    opt.Step();
    expect(AllClose(v->value, Tensor<2, 3>(0)));
  };

  auto f = [](const TVariable<Tensor<2, 1>>& v) {
    auto v2 = v * v;
    auto v3 = MatrixProduct(TVariable(Tensor<1, 2>({{1, 2}})), v2);
    return v3.View<>();
  };

  "momentum"_test = [f] {
    TVariable<Tensor<2, 1>> v({{1}, {3}}, true);
    MomentumOptimizerUnit opt(v, /* lr = */ .1, /* alpha = */ .9);

    f(v)->Backward();
    opt.Step();
    expect(AllClose(v->value, Tensor<2, 1>({{0.8}, {1.8}})));

    f(v)->Backward();
    opt.Step();
    expect(AllClose(v->value, Tensor<2, 1>({{0.46}, {0}})));
  };

  "adam"_test = [f] {
    TVariable<Tensor<2, 1>> v({{1}, {3}}, true);
    AdamOptimizerUnit opt(v, /* lr = */ .1, /* beta1 = */ .9, /* beta2 = */ 0.99, /* eps = */ 2e-2);

    f(v)->Backward();
    opt.Step();
    expect(AllClose(v->value, Tensor<2, 1>({{0.9009901}, {2.9001663}})));

    f(v)->Backward();
    opt.Step();
    expect(AllClose(v->value, Tensor<2, 1>({{0.8024092}, {2.8004302}})));
  };
};
