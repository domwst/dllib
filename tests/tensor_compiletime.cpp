#include <dllib/tensor.hpp>

template<size_t... Dims>
using Tensor = dllib::TTensor<int, Dims...>;

[[maybe_unused]] static consteval void StaticChecks() {
  {
    constexpr int data1[3][4] = {
      {4, 7, 1, 3},
      {9, 0, 8, 8},
      {3, 2, 6, 0},
    };
    constexpr int data2[3][4] = {
      {4, 2, 7, 2},
      {5, 4, 5, 3},
      {1, 0, 3, 6},
    };
    { // Addition test
      constexpr int sum[3][4] = {
        {8, 9, 8, 5},
        {14, 4, 13, 11},
        {4, 2, 9, 6},
      };
      static_assert(Tensor<3, 4>(data1) + Tensor<3, 4>(data2) == Tensor<3, 4>(sum));
    }
    { // Subtraction test
      constexpr int diff[3][4] = {
        {0, 5, -6, 1},
        {4, -4, 3, 5},
        {2, 2, 3, -6},
      };
      static_assert(Tensor<3, 4>(data1) - Tensor<3, 4>(data2) == Tensor<3, 4>(diff));
    }
    { // Matrix multiplication tests
      constexpr int data3[4][3] = {
        {4, 2, 7},
        {2, 5, 4},
        {5, 3, 1},
        {0, 3, 6},
      };
      constexpr int mul1[3][3] = {
        {35, 55, 75},
        {76, 66, 119},
        {46, 34, 35},
      };
      constexpr int mul2[4][4] = {
        {55, 42, 62, 28},
        {65, 22, 66, 46},
        {50, 37, 35, 39},
        {45, 12, 60, 24},
      };
      static_assert(dllib::MatrixProduct(Tensor<3, 4>(data1), Tensor<4, 3>(data3)) == Tensor<3, 3>(mul1));
      static_assert(dllib::MatrixProduct(Tensor<4, 3>(data3), Tensor<3, 4>(data1)) == Tensor<4, 4>(mul2));
    }
  }
  { // Sum tests
    constexpr int data[2][3][2] = {
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
    static_assert(Sum(Tensor<2, 3, 2>(data)) == 43);
  }
  { // ApplyFunction
    constexpr int data[2][3][2] = {
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
    constexpr Tensor<2, 3, 2> t(data);
    {
      constexpr int expected = 43;
      static_assert(dllib::ApplyFunction<3>(dllib::Sum<Tensor<2, 3, 2>>, t) == Tensor<>(expected));
    }
    {
      constexpr int expected[2] = {16, 27};
      static_assert(dllib::ApplyFunction<2>(dllib::Sum<Tensor<3, 2>>, t) == Tensor<2>(expected));
    }
    {
      constexpr int expected[2][3] = {
        {3, 7, 6},
        {9, 9, 9},
      };
      static_assert(dllib::ApplyFunction<1>(dllib::Sum<Tensor<2>>, t) == Tensor<2, 3>(expected));
    }
    {
      static_assert(dllib::ApplyFunction<0>(dllib::Sum<Tensor<>>, t) == t);
    }
    {
      constexpr auto SqLen = [](Tensor<2> v) -> int {
        int sm = 0;
        for (int i = 0; i < 2; ++i) {
          sm += v[i] * v[i];
        }
        return sm;
      };
      constexpr int expected[2][3] = {
        { 5, 25, 26},
        {81, 65, 53},
      };
      static_assert(dllib::ApplyFunction<1>(SqLen, t) == Tensor<2, 3>(expected));
    }
  }
}
