#include <dllib/autograd.hpp>
#include <boost/ut.hpp>

namespace ut = boost::ut;

static ut::suite autograd = [] {
  using namespace ut;
  using namespace dllib;

  "sum_all"_test = [] {
    {
      int data[2][3] = {
        {1, 2, 3},
        {4, 5, 6},
      };

      TVariable<TTensor<int, 2, 3>> v(data, true);

      auto sm = Sum(v);
      expect(eq(sm->value, TTensor<int>(21)));
      sm->Backward();
      expect(eq(v->grad, TTensor<int, 2, 3>(1)));
    }
    {
      int data[5] = {1, 2, 3, 4, 5};

      TVariable<TTensor<int, 5>> v(data, true);

      auto sm = Sum(v);
      expect(eq(sm->value, TTensor<int>(15)));
      sm->Backward();
      expect(eq(v->grad, TTensor<int, 5>(1)));
    }
  };
};