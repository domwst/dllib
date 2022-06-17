#include <dllib/autograd.hpp>
#include <dllib/serialization.hpp>

#include <boost/ut.hpp>

namespace ut = boost::ut;

static ut::suite autograd = [] {
  using namespace ut;
  using namespace dllib;

  "referencing"_test = [] {
    TVariable<TTensor<int, 2, 3>> v1({{1, 2, 3}, {4, 5, 6}}, true);
    auto v2 = v1;
    v2->value[0][1] += 1;
    expect(eq(v1->value, v2->value));
    auto v3 = v2.Copy();
    v3->value[1][2] += 1;
    expect(neq(v2->value, v3->value) && eq(v3->requires_grad, true));

    TVariable<TTensor<int, 2, 3>> v4({{1, 2, 3}, {3, 2, 1}}, false);
    auto v5 = v4.Copy();
    expect(eq(v4->requires_grad, false) && eq(v5->requires_grad, false));
  };

  "is_leaf"_test = [] {
    TVariable<TTensor<int, 2, 3>> v1({{1, 2, 3}, {4, 5, 6}}, true);
    expect(eq(v1.IsLeaf(), true));

    auto v2 = v1.Copy();
    expect(eq(v2.IsLeaf(), true));

    auto v3 = v1 + v2;
    expect(eq(v3.IsLeaf(), false));

    auto v4 = v3;
    expect(eq(v4.IsLeaf(), false));

    auto v5 = v4.Copy();
    expect(eq(v5.IsLeaf(), true));
  };

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

  "stack"_test = [] {
    TVariable<TTensor<int, 1, 2>> v1({{2, 0}}, true);
    TVariable<TTensor<int, 1, 2>> v2({{5, 4}}, true);

    auto v = StackAlong<0>(v1, v2);
    Sum(MatrixProduct(v, v))->Backward();

    TTensor<int, 1, 2> expected1 = {{9, 16}}, expected2 = {{6, 13}};
    expect(eq(v1->grad, expected1) && eq(v2->grad, expected2));
  };

  "serialization"_test = [] {
    std::stringstream ss;
    {
      TVariable<TTensor<int, 2, 3>> t;
      t->value = {{1, 2, 3}, {4, 5, 6}};
      t->grad = {{7, 8, 9}, {10, 11, 12}};
      Dump(ss, t);
    }
    {
      TVariable<TTensor<int, 2, 3>> t;
      Load(ss, t);
      expect(eq(t->value, TTensor<int, 2, 3>{{1, 2, 3}, {4, 5, 6}}) &&
             eq(t->grad, TTensor<int, 2, 3>{{7, 8, 9}, {10, 11, 12}}));
    }
  };

  "exp"_test = [] {
    TVariable<TTensor<float, 2, 2>> v({{1, 2}, {3, 4}}, true);
    auto exp = Exp(v);
    expect(eq(exp->value, Exp(v->value)));
    Sum(exp)->Backward();
    expect(eq(v->grad, exp->value));
  };

  "tanh"_test = [] {
    TVariable<TTensor<float, 2, 2>> v({{-1, -2}, {0, 1}}, true);
    auto tanh = Tanh(v);
    expect(eq(tanh->value, Tanh(v->value)));
    Sum(tanh)->Backward();
    TTensor<float, 2, 2> expected = {{0.41997466, 0.0706508}, {1, 0.4199740}};
    expect(AllClose(v->grad, expected));
  };

  "sigmoid"_test = [] {
    TVariable<TTensor<float, 2, 2>> v({{-1, -2}, {0, 1}}, true);
    auto sigm = Sigmoid(v);
    expect(eq(sigm->value, Sigmoid(v->value)));
    Sum(sigm)->Backward();
    TTensor<float, 2, 2> expected = {{0.19661197, 0.1049936}, {0.25, 0.19661197}};
    expect(AllClose(v->grad, expected));

  };
};
