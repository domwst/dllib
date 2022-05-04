#include <dllib/tensor.hpp>
#include <boost/ut.hpp>

namespace ut = boost::ut;

template<std::size_t... Dims>
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
      expect(eq(
        dllib::MatrixProduct(Tensor<3, 4>(data1), Tensor<4, 3>(data3)), Tensor<3, 3>(mul1)));
      expect(eq(
        dllib::MatrixProduct(Tensor<4, 3>(data3), Tensor<3, 4>(data1)), Tensor<4, 4>(mul2)));
    };
  }
  { // Tensor-matrix, tensor-vector multiplication tests
    int data1[2][3][2] = {
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
    "tensor_matrix_multiplication"_test = [data1] {
      int data2[2][3] = {
        {0, 2, 4},
        {0, 3, 9},
      };
      int expected[2][3][3] = {
        {
          {0, 8,  22},
          {0, 18, 48},
          {0, 28, 74},
        },
        {
          {0, 12, 30},
          {0, 20, 58},
          {0, 22, 56},
        },
      };
      expect(eq(dllib::MatrixProduct(Tensor<2, 3, 2>(data1), Tensor<2, 3>(data2)), Tensor<2, 3, 3>(expected)));
    };
    "matrix_vector_multiplication"_test = [data1] {
      int data2[2] = {1, 2};
      int expected[2][3] = {
        {5, 11, 17},
        {7, 13, 13},
      };
      expect(eq(dllib::MatrixProduct(Tensor<2, 3, 2>(data1), Tensor<2>(data2)), Tensor<2, 3>(expected)));
      expect(eq(dllib::MatrixProduct(Tensor<3, 2>(data1[0]), Tensor<2>(data2)), Tensor<3>(expected[0])));
    };
  }
  { // Vector-vector, vector-matrix multiplication tests
    int data1[3] = {1, 2, 3};
    "vector_vector_multiplication"_test = [data1] {
      int data2[3] = {3, 2, 1};
      int expected = 10;
      expect(eq(dllib::MatrixProduct(Tensor<3>(data1), Tensor<3>(data2)), Tensor<>(expected)));
    };
    "vector_matrix_multiplication"_test = [data1] {
      int data2[3][2] = {
        {3, 2},
        {2, 4},
        {3, 9},
      };
      int expected[2] = {16, 37};
      expect(eq(dllib::MatrixProduct(Tensor<3>(data1), Tensor<3, 2>(data2)), Tensor<2>(expected)));
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
};
