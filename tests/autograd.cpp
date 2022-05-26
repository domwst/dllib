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

  "sum"_test = [] {
    int data1[2][3] = {
      {1, 2, 3},
      {4, 5, 6},
    };
    TVariable<TTensor<int, 2, 3>> v1(data1, true);

    int data2[2][3] = {
      {7, 8, 9},
      {0, 2, 1},
    };
    TVariable<TTensor<int, 2, 3>> v2(data2, true);

    auto sum = v1 + v2;
    {
      int expected[2][3] = {
        {8, 10, 12},
        {4, 7,  7},
      };
      expect(eq(sum->value, TTensor<int, 2, 3>(expected)));
    }
    auto sum_all = Sum(sum);
    expect(eq(sum_all->value, TTensor<int>(48)));
    sum_all->Backward();
    expect(eq(v1->grad, TTensor<int, 2, 3>(1)));
    expect(eq(v2->grad, TTensor<int, 2, 3>(1)));
  };

  "difference"_test = [] {
    int data1[2][3] = {
      {1, 2, 3},
      {4, 5, 6},
    };
    TVariable<TTensor<int, 2, 3>> v1(data1, true);

    int data2[2][3] = {
      {7, 8, 9},
      {0, 2, 1},
    };
    TVariable<TTensor<int, 2, 3>> v2(data2, true);

    auto diff = v1 - v2;
    {
      int expected[2][3] = {
        {-6, -6, -6},
        { 4,  3,  5},
      };
      expect(eq(diff->value, TTensor<int, 2, 3>(expected)));
    }
    auto sum_all = Sum(diff);
    expect(eq(sum_all->value, TTensor<int>(-6)));
    sum_all->Backward();
    expect(eq(v1->grad, TTensor<int, 2, 3>(1)));
    expect(eq(v2->grad, TTensor<int, 2, 3>(-1)));
  };

  "multiplication"_test = [] {
    int data1[2][3] = {
      {1, 2, 3},
      {4, 5, 6},
    };
    TVariable<TTensor<int, 2, 3>> v1(data1, true);

    int data2[2][3] = {
      {7, 8, 9},
      {0, 2, 1},
    };
    TVariable<TTensor<int, 2, 3>> v2(data2, true);

    auto multiplication = v1 * v2;
    {
      int expected[2][3] = {
        { 7, 16, 27},
        { 0, 10,  6},
      };
      expect(eq(multiplication->value, TTensor<int, 2, 3>(expected)));
    }
    auto sum_all = Sum(multiplication);
    expect(eq(sum_all->value, TTensor<int>(66)));
    sum_all->Backward();
    expect(eq(v1->grad, TTensor<int, 2, 3>(data2)));
    expect(eq(v2->grad, TTensor<int, 2, 3>(data1)));
  };

  "matrix_product_1"_test = [] {
    int data1[2][3] = {
      {1, 2, 3},
      {4, 5, 6},
    };
    TVariable<TTensor<int, 2, 3>> v1(data1, true);

    int data2[3][2] = {
      {9, 8},
      {7, 6},
      {5, 4},
    };
    TVariable<TTensor<int, 3, 2>> v2(data2, true);

    auto prod = MatrixProduct(v1, v2);
    {
      int expected[2][2] = {
        { 38, 32},
        {101, 86},
      };
      expect(eq(prod->value, TTensor<int, 2, 2>(expected)));
    }

    Sum(prod)->Backward();

    {
      int expected[2][3] = {
        {17, 13, 9},
        {17, 13, 9},
      };
      expect(eq(v1->grad, TTensor<int, 2, 3>(expected)));
    }
    {
      int expected[3][2] = {
        {5, 5},
        {7, 7},
        {9, 9},
      };
      expect(eq(v2->grad, TTensor<int, 3, 2>(expected)));
    }
  };

  "view"_test = [] {
    int data[2][3][2] = {
      {
        {1, 2},
        {3, 4},
        {5, 6},
      },
      {
        {7, 8},
        {9, 0},
        {1, 2},
      },
    };
    TTensor<int, 2, 3, 2> t(data);
    TVariable v1(t, true);

    {
      auto v2 = v1.View<3, 4>();
      expect(eq(v2->value, t.View<3, 4>()));
      Sum(v2)->Backward();
    }
    expect(eq(v1->grad, TTensor<int, 2, 3, 2>(1)));

    {
      auto v2 = v1.View<12>();
      expect(eq(v2->value, t.View<12>()));
      Sum(v2)->Backward();
    }
    expect(eq(v1->grad, TTensor<int, 2, 3, 2>(2)));
  };

  "self_sum"_test = [] {
    int data[3] = {1, 2, 3};
    TVariable<TTensor<int, 3>> v(data, true);

    auto sm = v + v;
    {
      int expected[3] = {2, 4, 6};
      expect(eq(sm->value, TTensor<int, 3>(expected)));
    }
    Sum(sm)->Backward();

    expect(eq(v->grad, TTensor<int, 3>(2)));
  };

  "complex_chaining_1"_test = [] {
    int data1[2][2] = {
      {1, 2},
      {3, 4},
    };
    TVariable<TTensor<int, 2, 2>> v1(data1, true);

    int data2[2][2] = {
      {4, 5},
      {2, 3},
    };
    TVariable<TTensor<int, 2, 2>> v2(data2, true);

    Sum(MatrixProduct(v1 - v2, v1 + v2 + v2))->Backward();
    {
      int expected[2][2] = {
        {19, 15},
        {19, 15},
      };
      expect(eq(v1->grad, TTensor<int, 2, 2>(expected)));
    }
    {
      int expected[2][2] = {
        {-25, -21},
        {-25, -21},
      };
      expect(eq(v2->grad, TTensor<int, 2, 2>(expected)));
    }
  };

  "proper_requires_grad_propagation"_test = [] {
    int data1[2][2] = {
      {1, 2},
      {3, 4},
    };
    TVariable<TTensor<int, 2, 2>> v1(data1, true);

    int data2[2][2] = {
      {4, 5},
      {2, 3},
    };
    TVariable<TTensor<int, 2, 2>> v2(data2, false);

    Sum(MatrixProduct(v1 - v2, v1 + v2 + v2))->Backward();
    {
      int expected[2][2] = {
        {19, 15},
        {19, 15},
      };
      expect(eq(v1->grad, TTensor<int, 2, 2>(expected)));
    }
    expect(eq(v2->grad, TTensor<int, 2, 2>(0)));
  };

  "transpose"_test = [] {
    int data[2][2] = {
      {1, 2},
      {3, 4},
    };
    TVariable<TTensor<int, 2, 2>> v(data, true);
    Sum(MatrixProduct(v, v.T()))->Backward();

    int expected[2][2] = {
      {8, 12},
      {8, 12},
    };
    expect(eq(v->grad, TTensor<int, 2, 2>(expected)));
  };

  "sqrt"_test = [] {
    TVariable<TTensor<float, 2, 3>> v({
      {1, 2, 3},
      {4, 5, 6},
    }, true);

    Sum(Sqrt(v))->Backward();
    TTensor<float, 2, 3> expected = {
      {0.5,  0.3535533, 0.2886751},
      {0.25, 0.2236067, 0.2041241},
    };
    expect(AllClose(v->grad, expected));
  };

  "log"_test = [] {
    TVariable<TTensor<float, 2, 3>> v({
      {1, 2, 3},
      {4, 5, 6},
    }, true);

    Sum(Log(v))->Backward();
    TTensor<float, 2, 3> expected = {
      {1. / 1, 1. / 2, 1. / 3},
      {1. / 4, 1. / 5, 1. / 6},
    };
    expect(AllClose(v->grad, expected));
  };
};
