#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <type_traits>
#include <utility>
#include <string_view>
#include <ostream>
#include <cmath>
#include <tuple>

namespace dllib {

template<class TData, size_t... Dims>
class TTensor;

namespace helpers {

template<class>
struct TIsTensorHelper : std::false_type {
};

template<class TData, size_t... Dims>
struct TIsTensorHelper<TTensor<TData, Dims...>> : std::true_type {
};

template<class, class>
struct TIsTensorOfTypeHelper : std::false_type {
};

template<class TData, size_t... Dims>
struct TIsTensorOfTypeHelper<TTensor<TData, Dims...>, TData> : std::true_type {
};

template<class, size_t...>
struct TIsTensorWithDimsHelper : std::false_type {
};

template<class TData, size_t... Dims>
struct TIsTensorWithDimsHelper<TTensor<TData, Dims...>, Dims...> : std::true_type {
};

}  // namespace helpers

template<class T>
constexpr bool VIsTensor = helpers::TIsTensorHelper<std::remove_cvref_t<T>>::value;

template<class T>
concept CTensor = VIsTensor<T>;


namespace helpers {

template<class T>
concept CHasBeginEnd = requires(const T& value) {
  { std::begin(value) };
  { std::end(value) };
};


template<class, std::array, class>
struct MakeTensorHelper {
};

template<class TData, size_t N, std::array<size_t, N> Dims, size_t... I>
struct MakeTensorHelper<TData, Dims, std::index_sequence<I...>> {
  using type = TTensor<TData, Dims[I]...>;
};

template<class, std::array>
struct MakeTensor {
};

template<class TData, size_t N, std::array<size_t, N> Dims>
struct MakeTensor<TData, Dims> {
  using type = typename MakeTensorHelper<TData, Dims, std::make_index_sequence<N>>::type;
};

template<class TData, std::array Dims>
using TMakeTensor = typename MakeTensor<TData, Dims>::type;

template<class, class>
struct MatrixProductResult {
};

template<class TData, size_t Dim1, size_t Dim2, size_t Dim3>
struct MatrixProductResult<TTensor<TData, Dim1, Dim2>, TTensor<TData, Dim2, Dim3>> {
  using type = TTensor<TData, Dim1, Dim3>;
};

template<class T1, class T2>
using TMatrixProductResult = typename MatrixProductResult<T1, T2>::type;

template<class>
struct TransposeResult {
};

template<class TData, size_t Dim1, size_t Dim2>
struct TransposeResult<TTensor<TData, Dim1, Dim2>> {
  using type = TTensor<TData, Dim2, Dim1>;
};

template<class T>
using TTransposeResult = typename TransposeResult<T>::type;

template<size_t DimsToSkip, class TFunction, CTensor... TArgs>
struct ApplyFunctionResult {
  static consteval auto GetDims() {
    using TFirstArg = std::tuple_element_t<0, std::tuple<TArgs...>>;

    std::array<size_t, DimsToSkip + TensorInfo<TRetData>::DimensionCount> ret{};
    std::copy(TFirstArg::Dimensions.begin(), TFirstArg::Dimensions.begin() + DimsToSkip, ret.begin());
    if constexpr (VIsTensor<TFunctionRet>) {
      std::copy(TFunctionRet::Dimensions.begin(), TFunctionRet::Dimensions.end(), ret.begin() + DimsToSkip);
    }
    return ret;
  }

  template<class T>
  struct TensorInfo {
    using TData = T;
    static constexpr size_t DimensionCount = 0;
  };

  template<CTensor T>
  struct TensorInfo<T> {
    using TData = typename T::TData;
    static constexpr size_t DimensionCount = T::DimensionCount;
  };

  using TFunctionRet = std::invoke_result_t<
    TFunction,
    typename TArgs::template TSubTensor<TArgs::DimensionCount - DimsToSkip>...>;

  using TRetData = std::conditional_t<VIsTensor<TFunctionRet>, typename TensorInfo<TFunctionRet>::TData, TFunctionRet>;
  using type = TMakeTensor<TRetData, GetDims()>;
};

template<size_t DimsToSkip, class TFunction, CTensor... TArgs>
using TApplyFunctionResult = typename ApplyFunctionResult<DimsToSkip, TFunction, TArgs...>::type;

}  // namespace helpers

template<class TData, std::array Dimensions>
using TMakeTensor = helpers::TMakeTensor<TData, Dimensions>;

template<class TData>
class TTensor<TData> {
 public:

  using DataType = TData;
  static constexpr size_t TotalElements = 1;
  static constexpr size_t DimensionCount = 0;
  static constexpr std::array<size_t, 0> Dimensions{};

  template<size_t N>
  using TSubTensor = std::conditional_t<N == 0, TTensor, void>;


  constexpr TTensor() = default;

  // NOLINTNEXTLINE
  constexpr TTensor(TData val) {
    FillWith(val);
  }

  constexpr TTensor& operator=(TData val) {
    data_ = val;
    return *this;
  }

  template<size_t... NewDims>
  const TTensor<TData, NewDims...>& View() const {
    static_assert(TTensor<TData, NewDims...>::TotalElements == TotalElements);
    return *reinterpret_cast<const TTensor<TData, NewDims...>*>(this);
  }

  template<size_t... NewDims>
  TTensor<TData, NewDims...>& View() {
    static_assert(TTensor<TData, NewDims...>::TotalElements == TotalElements);
    return *reinterpret_cast<TTensor<TData, NewDims...>*>(this);
  }

  constexpr TTensor& FillWith(TData val) {
    return (*this) = val;
  }

  template<class TOtherData>
  constexpr auto To() const {
    return TMakeTensor<TOtherData, Dimensions>(Data());
  }

  constexpr TData Data() const {
    return data_;
  }

  constexpr TData& Data() {
    return data_;
  }

  constexpr operator TData&() {
    return data_;
  }

  constexpr operator const TData&() const {
    return data_;
  }

  constexpr bool operator==(const TTensor&) const = default;

 private:
  TData data_;
};

template<class TData, size_t FirstDim, size_t... OtherDims>
class TTensor<TData, FirstDim, OtherDims...> {
 public:

  using DataType = TData;
  using ElementType = TTensor<TData, OtherDims...>;
  using ContainerType = std::array<ElementType, FirstDim>;

  static constexpr size_t TotalElements = ElementType::TotalElements * FirstDim;
  static constexpr size_t DimensionCount = sizeof...(OtherDims) + 1;
  static constexpr std::array<size_t, DimensionCount> Dimensions = {FirstDim, OtherDims...};

  template<size_t N>
  using TSubTensor = std::conditional_t<N == DimensionCount, TTensor, typename ElementType::template TSubTensor<N>>;

  constexpr TTensor() = default;

  constexpr explicit TTensor(TData val) {
    FillWith(val);
  }

  constexpr TTensor(const std::initializer_list<ElementType>& init) : TTensor(init.begin(), init.end()) {
  }

  template<class TForwardIt>
  constexpr TTensor(TForwardIt begin, TForwardIt end) {
    FillWith(begin, end);
  }

  template<helpers::CHasBeginEnd T>
  constexpr TTensor& operator=(const T& value) {
    FillWith(std::begin(value), std::end(value));
    return *this;
  }

  template<helpers::CHasBeginEnd T>
  constexpr TTensor(const T& value) : TTensor(std::begin(value), std::end(value)) {}

  constexpr TTensor& operator+=(const TTensor& other) {
    for (size_t i = 0; i < FirstDim; ++i) {
      data_[i] += other[i];
    }
    return *this;
  }

  constexpr TTensor& operator+=(TData val) {
    for (size_t i = 0; i < FirstDim; ++i) {
      data_[i] += val;
    }
    return *this;
  }

  constexpr TTensor operator+(const TTensor& other) const {
    return TTensor(*this) += other;
  }

  constexpr TTensor operator+(TData val) const {
    return TTensor(*this) += val;
  }

  constexpr TTensor& operator-=(const TTensor& other) {
    for (size_t i = 0; i < FirstDim; ++i) {
      data_[i] -= other[i];
    }
    return *this;
  }

  constexpr TTensor& operator-=(TData val) {
    for (size_t i = 0; i < FirstDim; ++i) {
      data_[i] -= val;
    }
    return *this;
  }

  constexpr TTensor operator-(const TTensor& other) const {
    return TTensor(*this) -= other;
  }

  constexpr TTensor operator-(TData val) const {
    return TTensor(*this) -= val;
  }

  constexpr TTensor& operator*=(const TTensor& other) {
    for (size_t i = 0; i < FirstDim; ++i) {
      data_[i] *= other[i];
    }
    return *this;
  }

  constexpr TTensor& operator*=(TData val) {
    for (size_t i = 0; i < FirstDim; ++i) {
      data_[i] *= val;
    }
    return *this;
  }

  constexpr TTensor operator*(const TTensor& other) const {
    return TTensor(*this) *= other;
  }

  constexpr TTensor operator*(TData val) const {
    return TTensor(*this) *= val;
  }

  constexpr TTensor& operator/=(const TTensor& other) {
    for (size_t i = 0; i < FirstDim; ++i) {
      data_[i] /= other[i];
    }
    return *this;
  }

  constexpr TTensor& operator/=(TData val) {
    for (size_t i = 0; i < FirstDim; ++i) {
      data_[i] /= val;
    }
    return *this;
  }

  constexpr TTensor operator/(const TTensor& other) const {
    return TTensor(*this) /= other;
  }

  constexpr TTensor operator/(TData val) const {
    return TTensor(*this) /= val;
  }

  constexpr TTensor operator-() const {
    return TTensor(*this) *= TData(-1);
  }

  constexpr const ElementType& operator[](size_t idx) const {
    return data_[idx];
  }

  constexpr ElementType& operator[](size_t idx) {
    return data_[idx];
  }

  template<size_t... NewDims>
  const TTensor<TData, NewDims...>& View() const {
    static_assert(TTensor<TData, NewDims...>::TotalElements == TotalElements);
    return *reinterpret_cast<const TTensor<TData, NewDims...>*>(this);
  }

  template<size_t... NewDims>
  TTensor<TData, NewDims...>& View() {
    static_assert(TTensor<TData, NewDims...>::TotalElements == TotalElements);
    return *reinterpret_cast<TTensor<TData, NewDims...>*>(this);
  }

  constexpr TTensor& FillWith(TData val) {
    for (size_t i = 0; i < FirstDim; ++i) {
      data_[i].FillWith(val);
    }
    return *this;
  }

  template<class TForwardIt>
  constexpr TTensor& FillWith(TForwardIt begin, TForwardIt end) {
    assert(std::distance(begin, end) == FirstDim);
    for (size_t i = 0; i < FirstDim; ++i) {
      data_[i] = *begin;
      ++begin;
    }
    return *this;
  }

  constexpr const ContainerType& Data() const {
    return data_;
  }

  constexpr auto begin() const {
    return data_.begin();
  }

  constexpr auto end() const {
    return data_.end();
  }

  constexpr auto begin() {
    return data_.begin();
  }

  constexpr auto end() {
    return data_.end();
  }

  template<class T = TTensor>
  constexpr auto T() const { // FIX HERE
    helpers::TTransposeResult<T> result;
    for (size_t i = 0; i < Size(); ++i) {
      for (size_t j = 0; j < data_[i].Size(); ++j) {
        result[j][i] = data_[i][j];
      }
    }
    return result;
  }

  template<class TOtherData>
  constexpr auto To() const {
    return TMakeTensor<TOtherData, Dimensions>(begin(), end());
  }

  constexpr static size_t Size() {
    return FirstDim;
  }

  constexpr bool operator==(const TTensor&) const = default;

 private:
  ContainerType data_;
};

template<CTensor Tensor>
constexpr Tensor operator+(typename Tensor::DataType val, const Tensor& other) {
  return Tensor(other) += val;
}

template<CTensor Tensor>
constexpr Tensor operator-(typename Tensor::DataType val, const Tensor& other) {
  return (Tensor(other) -= val) *= typename Tensor::DataType(-1);
}

template<CTensor Tensor>
constexpr Tensor operator*(typename Tensor::DataType val, const Tensor& other) {
  return Tensor(other) *= val;
}

template<size_t DimsToSkip, class TFunction, CTensor Tensor>
constexpr void ApplyFunctionInplace(TFunction&& function, Tensor& tensor) {
  static_assert(Tensor::DimensionCount >= DimsToSkip);
  if constexpr (Tensor::DimensionCount == DimsToSkip) {
    tensor = function(tensor.Data());
  } else {
    for (size_t i = 0; i < tensor.Size(); ++i) {
      ApplyFunctionInplace<DimsToSkip>(function, tensor[i]);
    }
  }
}

template<size_t DimsToSkip, class TFunction, class TResult, CTensor... TArgs>
constexpr void ApplyFunctionTo(TFunction&& function, TResult& result, const TArgs&... args) {
  []<size_t... i>(std::index_sequence<i...>) {
    static_assert(((DimsToSkip <= std::tuple_element_t<i, std::tuple<TArgs...>>::DimensionCount) && ...));
  }(std::make_index_sequence<sizeof...(TArgs)>{});
  if constexpr (DimsToSkip == 0) {
    result = function(args.Data()...);
  } else {
    for (size_t i = 0; i < result.Size(); ++i) {
      ApplyFunctionTo<DimsToSkip - 1>(function, result[i], args[i]...);
    }
  }
}

template<size_t DimsToSkip, class TFunction, CTensor... TArgs>
constexpr helpers::TApplyFunctionResult<DimsToSkip, TFunction, TArgs...>
  ApplyFunction(TFunction&& function, const TArgs&... args) {

  helpers::TApplyFunctionResult<DimsToSkip, TFunction, TArgs...> result;
  ApplyFunctionTo<DimsToSkip>(std::forward<TFunction>(function), result, args...);
  return result;
}

template<class TData, size_t Dim1, size_t Dim2, size_t Dim3>
void MatrixProduct(
  const TTensor<TData, Dim1, Dim2>& matrix1,
  const TTensor<TData, Dim2, Dim3>& matrix2,
  TTensor<TData, Dim1, Dim3>& result) {

  for (size_t i = 0; i < Dim1; ++i) {
    for (size_t j = 0; j < Dim2; ++j) {
      for (size_t k = 0; k < Dim3; ++k) {
        result[i][k] += matrix1[i][j] * matrix2[j][k];
      }
    }
  }
}

template<class TData, size_t Dim1, size_t Dim2, size_t Dim3>
TTensor<TData, Dim1, Dim3> MatrixProduct(
  const TTensor<TData, Dim1, Dim2>& matrix1,
  const TTensor<TData, Dim2, Dim3>& matrix2) {

  TTensor<TData, Dim1, Dim3> result(0);
  MatrixProduct(matrix1, matrix2, result);
  return result;
}

template<class TData, size_t Dim1, size_t Dim2, size_t Dim3>
void MatrixProductTransposed(
  const TTensor<TData, Dim1, Dim2>& matrix1,
  const TTensor<TData, Dim3, Dim2>& matrix2_T,
  TTensor<TData, Dim1, Dim3>& result) {

  for (size_t i = 0; i < Dim1; ++i) {
    for (size_t j = 0; j < Dim3; ++j) {
      for (size_t k = 0; k < Dim2; ++k) {
        result[i][j] += matrix1[i][k] * matrix2_T[j][k];
      }
    }
  }
}

template<class TData, size_t Dim1, size_t Dim2, size_t Dim3>
TTensor<TData, Dim1, Dim3> MatrixProductTransposed(
  const TTensor<TData, Dim1, Dim2>& matrix1,
  const TTensor<TData, Dim3, Dim2>& matrix2_T) {

  TTensor<TData, Dim1, Dim3> result;
  MatrixProductTransposed(matrix1, matrix2_T, result);
  return result;
}

template<CTensor T>
T Sqrt(T inp) {
  ApplyFunctionInplace<0>(static_cast<typename T::DataType (*)(typename T::DataType)>(std::sqrt), inp);
  return inp;
}

template<CTensor T>
T Log(T inp) {
  ApplyFunctionInplace<0>(static_cast<typename T::DataType (*)(typename T::DataType)>(std::log), inp);
  return inp;
}

template<CTensor T>
constexpr typename T::DataType Sum(const T& arg) {
  if constexpr (T::DimensionCount == 0) {
    return arg;
  } else {
    typename T::DataType sm = 0;
    for (size_t i = 0; i < arg.Size(); ++i) {
      sm += Sum(arg[i]);
    }
    return sm;
  }
}

template<CTensor T>
std::ostream& operator<<(std::ostream& out, T tensor) {
  if constexpr (T::DimensionCount > 0) {
    out << "Tensor<";
    {
      bool first = true;
      for (auto x: T::Dimensions) {
        if (!first) {
          out << ", ";
        }
        first = false;
        out << x;
      }
    }
    out << '>';
  }
  {
    bool first = true;
    out << "{";
    for (auto x : tensor.template View<T::TotalElements>()) {
      if (!first) {
        out << ", ";
      }
      first = false;
      out << x.Data();
    }
  }
  out << "}";
  return out;
}

}  // namespace dllib
