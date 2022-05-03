#include <dllib/Tensor.hpp>

template<std::size_t... Dims>
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
      static_assert(dllib::MatrixMultiplication(Tensor<3, 4>(data1), Tensor<4, 3>(data3)) == Tensor<3, 3>(mul1));
      static_assert(dllib::MatrixMultiplication(Tensor<4, 3>(data3), Tensor<3, 4>(data1)) == Tensor<4, 4>(mul2));
    }
  }
  { // Tensor-matrix, tensor-vector multiplication tests
    constexpr int data1[2][3][2] = {
      {
        {1, 2},
        {3, 4},
        {5, 6},
      },
      {
        {3, 2},
        {1, 6},
        {5, 4},
      },
    };
    {
      constexpr int data2[2][3] = {
        {0, 2, 4},
        {0, 3, 9},
      };
      constexpr int expected[2][3][3] = {
        {
          {0,  8, 22},
          {0, 18, 48},
          {0, 28, 74},
        },
        {
          {0, 12, 30},
          {0, 20, 58},
          {0, 22, 56},
        },
      };
      static_assert(dllib::MatrixMultiplication(Tensor<2, 3, 2>(data1), Tensor<2, 3>(data2)) == Tensor<2, 3, 3>(expected));
    }
    {
      constexpr int data2[2] = {1, 2};
      constexpr int expected[2][3] = {
        {5, 11, 17},
        {7, 13, 13},
      };
      static_assert(dllib::MatrixMultiplication(Tensor<2, 3, 2>(data1), Tensor<2>(data2)) == Tensor<2, 3>(expected));
      static_assert(dllib::MatrixMultiplication(Tensor<3, 2>(data1[0]), Tensor<2>(data2)) == Tensor<3>(expected[0]));
    }
  }
  { // Vector-vector, vector-matrix multiplication tests
    constexpr int data1[3] = {1, 2, 3};
    {
      constexpr int data2[3] = {3, 2, 1};
      constexpr int expected = 10;
      static_assert(dllib::MatrixMultiplication(Tensor<3>(data1), Tensor<3>(data2)) == Tensor<>(expected));
    }
    {
      constexpr int data2[3][2] = {
        {3, 2},
        {2, 4},
        {3, 9},
      };
      constexpr int expected[2] = {16, 37};
      static_assert(dllib::MatrixMultiplication(Tensor<3>(data1), Tensor<3, 2>(data2)) == Tensor<2>(expected));
    }
  }
  { // Matrix transpose tests
    constexpr int data[2][3] = {
      {1, 2, 3},
      {4, 5, 6},
    };
    constexpr int data_t[3][2] = {
      {1, 4},
      {2, 5},
      {3, 6},
    };
    static_assert(Tensor<2, 3>(data).T() == Tensor<3, 2>(data_t));
    static_assert(Tensor<3, 2>(data_t).T() == Tensor<2, 3>(data));
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
}
