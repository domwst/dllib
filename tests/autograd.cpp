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
      auto v2 = View<3, 4>(v1);
      expect(eq(v2->value, t.View<3, 4>()));
      Sum(v2)->Backward();
    }
    expect(eq(v1->grad, TTensor<int, 2, 3, 2>(1)));

    {
      auto v2 = View<12>(v1);
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
};
