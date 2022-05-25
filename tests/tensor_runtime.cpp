#include <dllib/tensor.hpp>
#include <boost/ut.hpp>

namespace ut = boost::ut;

template<size_t... Dims>
using Tensor = dllib::TTensor<int, Dims...>;

static ut::suite tensor_runtime_tests = [] {
  using namespace ut;
  {
    int data1[3][4] = {
      {4, 7, 1, 3},
      {9, 0, 8, 8},
      {3, 2, 6, 0},
    };
    int data2[3][4] = {
      {4, 2, 7, 2},
      {5, 4, 5, 3},
      {1, 0, 3, 6},
    };
    "addition"_test = [data1, data2] { // Addition test
      constexpr int sum[3][4] = {
        {8,  9, 8,  5},
        {14, 4, 13, 11},
        {4,  2, 9,  6},
      };
      expect(eq(Tensor<3, 4>(data1) + Tensor<3, 4>(data2), Tensor<3, 4>(sum)));
    };
    "subtraction"_test = [data1, data2] { // Subtraction test
      int diff[3][4] = {
        {0, 5,  -6, 1},
        {4, -4, 3,  5},
        {2, 2,  3,  -6},
      };
      expect(eq(Tensor<3, 4>(data1) - Tensor<3, 4>(data2), Tensor<3, 4>(diff)));
    };
    "matrix_multiplication"_test = [data1] { // Matrix multiplication tests
      int data3[4][3] = {
        {4, 2, 7},
        {2, 5, 4},
        {5, 3, 1},
        {0, 3, 6},
      };
      int mul1[3][3] = {
        {35, 55, 75},
        {76, 66, 119},
        {46, 34, 35},
      };
      int mul2[4][4] = {
        {55, 42, 62, 28},
        {65, 22, 66, 46},
        {50, 37, 35, 39},
        {45, 12, 60, 24},
      };
      expect(eq(dllib::MatrixProduct(Tensor<3, 4>(data1), Tensor<4, 3>(data3)), Tensor<3, 3>(mul1)));
      expect(eq(dllib::MatrixProduct(Tensor<4, 3>(data3), Tensor<3, 4>(data1)), Tensor<4, 4>(mul2)));
      expect(eq(dllib::MatrixProductTransposed(Tensor<3, 4>(data1), Tensor<4, 3>(data3).T()), Tensor<3, 3>(mul1)));
      expect(eq(dllib::MatrixProductTransposed(Tensor<4, 3>(data3), Tensor<3, 4>(data1).T()), Tensor<4, 4>(mul2)));
    };
  }
  "matrix_transpose"_test = [] { // Matrix transpose tests
    int data[2][3] = {
      {1, 2, 3},
      {4, 5, 6},
    };
    int data_t[3][2] = {
      {1, 4},
      {2, 5},
      {3, 6},
    };
    expect(eq(Tensor<2, 3>(data).T(), Tensor<3, 2>(data_t)));
    expect(eq(Tensor<3, 2>(data_t).T(), Tensor<2, 3>(data)));
  };
  "sum"_test = [] { // Sum tests
    int data[2][3][2] = {
      {
        {1, 2},
        {3, 4},
        {5, 1},
      },
      {
        {0, 9},
        {1, 8},
        {2, 7},
      },
    };
    expect(eq(Sum(Tensor<2, 3, 2>(data)), 43));
  };
  "view"_test = [] {
    int data[2][3][2] = {
      {
        {1, 2},
        {3, 4},
        {5, 1},
      },
      {
        {0, 9},
        {1, 8},
        {2, 7},
      },
    };
    Tensor<2, 3, 2> v1(data);

    {
      int expected[12] = {1, 2, 3, 4, 5, 1, 0, 9, 1, 8, 2, 7};
      Tensor<12> v2(expected);
      expect(eq(v1.View<12>(), v2));
      expect(eq(v2.View<2, 3, 2>(), v1));
    }
    {
      int expected[3][4] = {
        {1, 2, 3, 4},
        {5, 1, 0, 9},
        {1, 8, 2, 7},
      };
      Tensor<3, 4> v2(expected);
      expect(eq(v1.View<3, 4>(), v2));
      expect(eq(v2.View<2, 3, 2>(), v1));
    }
  };
  "apply_function"_test = [] {
    int data[2][3][2] = {
      {
        {1, 2},
        {3, 4},
        {5, 1},
      },
      {
        {0, 9},
        {1, 8},
        {2, 7},
      },
    };
    Tensor<2, 3, 2> t(data);
    {
      int expected = 43;
      expect(eq(dllib::ApplyFunction<0>(dllib::Sum<Tensor<2, 3, 2>>, t), Tensor<>(expected)));
    }
    {
      int expected[2] = {16, 27};
      expect(eq(dllib::ApplyFunction<1>(dllib::Sum<Tensor<3, 2>>, t), Tensor<2>(expected)));
    }
    {
      int expected[2][3] = {
        {3, 7, 6},
        {9, 9, 9},
      };
      expect(eq(dllib::ApplyFunction<2>(dllib::Sum<Tensor<2>>, t), Tensor<2, 3>(expected)));
    }
    {
      expect(eq(dllib::ApplyFunction<3>(dllib::Sum<Tensor<>>, t), t));
    }
    {
      auto SqLen = [](Tensor<2> v) -> int {
        int sm = 0;
        for (int i = 0; i < 2; ++i) {
          sm += v[i] * v[i];
        }
        return sm;
      };
      int expected[2][3] = {
        { 5, 25, 26},
        {81, 65, 53},
      };
      expect(eq(dllib::ApplyFunction<2>(SqLen, t), Tensor<2, 3>(expected)));
    }
  };
  "apply_function_with_multiple_arguments"_test = [] {
    int data1[2][3][2] = {
      {
        {1, 2},
        {3, 4},
        {5, 1},
      },
      {
        {0, 9},
        {1, 8},
        {2, 7},
      },
    };
    int data2[2][3][2] = {
      {
        {2, 7},
        {5, 1},
        {0, 9},
      },
      {
        {1, 8},
        {3, 4},
        {1, 2},
      },
    };
    int expected[2][3] = {
      {16, 19,  9},
      {72, 35, 16},
    };
    auto scalar_product = [](const Tensor<2>& a, const Tensor<2>& b) {
      return Sum(a * b);
    };
    expect(eq(
      ApplyFunction<2>(scalar_product, Tensor<2, 3, 2>(data1), Tensor<2, 3, 2>(data2)),
      Tensor<2, 3>(expected)));
  };
};
