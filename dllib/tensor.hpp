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

template<class TDataType, size_t... Dims>
class TTensor;

namespace helpers {

template<class>
struct TIsTensorHelper : std::false_type {
};

template<class TDataType, size_t... Dims>
struct TIsTensorHelper<TTensor<TDataType, Dims...>> : std::true_type {
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

template<class TDataType, size_t N, std::array<size_t, N> Dims, size_t... I>
struct MakeTensorHelper<TDataType, Dims, std::index_sequence<I...>> {
  using type = TTensor<TDataType, Dims[I]...>;
};

template<class, std::array>
struct MakeTensor {
};

template<class TDataType, size_t N, std::array<size_t, N> Dims>
struct MakeTensor<TDataType, Dims> {
  using type = typename MakeTensorHelper<TDataType, Dims, std::make_index_sequence<N>>::type;
};

template<class TDataType, std::array Dims>
using TMakeTensor = typename MakeTensor<TDataType, Dims>::type;

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

template<size_t Dim, CTensor T1, CTensor T2>
struct StackAlongResult {
  static consteval std::array<size_t, T1::DimensionCount> GetDims() {
    static_assert(std::is_same_v<typename T1::TData, typename T2::TData>);
    static_assert(T1:: DimensionCount == T2::DimensionCount);

    std::array<size_t, T1::DimensionCount> result = T1::Dimensions;
    result[Dim] += T2::Dimensions[Dim];

    // TODO Static check for dimensions

    return result;
  }

  using type = TMakeTensor<typename T1::TData, GetDims()>;
};

template<size_t Dim, CTensor T1, CTensor T2>
using TStackAlongResult = typename StackAlongResult<Dim, T1, T2>::type;

template<size_t Dim, size_t Size, CTensor T>
struct SplitAlongResult {
  static consteval std::array<std::array<size_t, T::DimensionCount>, 2> GetDims() {
    constexpr auto dims = T::Dimensions;
    static_assert(dims[Dim] > Size);

    auto first_dims = dims;
    first_dims[Dim] = Size;
    auto second_dims = dims;
    second_dims[Dim] -= Size;

    return {first_dims, second_dims};
  }

  using type = std::pair<
    TMakeTensor<typename T::TData, GetDims()[0]>,
    TMakeTensor<typename T::TData, GetDims()[1]>>;
};

template<size_t Dim, size_t Size, CTensor T>
using TSplitAlongResult = typename SplitAlongResult<Dim, Size, T>::type;

}  // namespace helpers

template<class TData, std::array Dimensions>
using TMakeTensor = helpers::TMakeTensor<TData, Dimensions>;

template<class TDataType>
class TTensor<TDataType> {
 public:

  using TData = TDataType;
  static constexpr size_t TotalElements = 1;
  static constexpr size_t DimensionCount = 0;
  static constexpr std::array<size_t, 0> Dimensions{};

  template<size_t N>
  using TSubTensor = std::conditional_t<N == 0, TTensor, void>;


  constexpr TTensor() = default;

  // NOLINTNEXTLINE
  constexpr TTensor(TDataType val) {
    FillWith(val);
  }

  constexpr TTensor& operator=(TDataType val) {
    data_ = val;
    return *this;
  }

  template<size_t... NewDims>
  const TTensor<TDataType, NewDims...>& View() const {
    static_assert(TTensor<TDataType, NewDims...>::TotalElements == TotalElements);
    return *reinterpret_cast<const TTensor<TDataType, NewDims...>*>(this);
  }

  template<size_t... NewDims>
  TTensor<TDataType, NewDims...>& View() {
    static_assert(TTensor<TDataType, NewDims...>::TotalElements == TotalElements);
    return *reinterpret_cast<TTensor<TDataType, NewDims...>*>(this);
  }

  constexpr TTensor& FillWith(TDataType val) {
    return (*this) = val;
  }

  template<class TOtherData>
  constexpr auto To() const {
    return TMakeTensor<TOtherData, Dimensions>(Data());
  }

  constexpr TDataType Data() const {
    return data_;
  }

  constexpr TDataType& Data() {
    return data_;
  }

  constexpr operator TDataType&() {
    return data_;
  }

  constexpr operator const TDataType&() const {
    return data_;
  }

  constexpr bool operator==(const TTensor&) const = default;

 private:
  TDataType data_;
};

template<class TDataType, size_t FirstDim, size_t... OtherDims>
class TTensor<TDataType, FirstDim, OtherDims...> {
 public:

  using TData = TDataType;
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

  template<class U = TTensor>
  constexpr std::enable_if_t<U::DimensionCount == 2, helpers::TTransposeResult<U>> T() const {
    helpers::TTransposeResult<U> result;
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
  constexpr bool operator!=(const TTensor&) const = default;

 private:
  ContainerType data_;
};

template<CTensor Tensor>
constexpr Tensor operator+(typename Tensor::TData val, const Tensor& other) {
  return Tensor(other) += val;
}

template<CTensor Tensor>
constexpr Tensor operator-(typename Tensor::TData val, const Tensor& other) {
  return (Tensor(other) -= val) *= typename Tensor::TData(-1);
}

template<CTensor Tensor>
constexpr Tensor operator*(typename Tensor::TData val, const Tensor& other) {
  return Tensor(other) *= val;
}

template<size_t DimsToSkip, class TFunction, CTensor Tensor>
constexpr void ApplyFunctionInplace(TFunction&& function, Tensor& tensor) {
  static_assert(Tensor::DimensionCount >= DimsToSkip);
  if constexpr (DimsToSkip == 0) {
    tensor = function(tensor.Data());
  } else {
    for (size_t i = 0; i < tensor.Size(); ++i) {
      ApplyFunctionInplace<DimsToSkip - 1>(function, tensor[i]);
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

template<class TDataType, size_t... Dims>
constexpr TTensor<bool, Dims...> operator<(
  const TTensor<TDataType, Dims...>& t1,
  const TTensor<TDataType, Dims...>& t2) {

  return ApplyFunction<sizeof...(Dims)>([](TDataType v1, TDataType v2) {
    return v1 < v2;
  }, t1, t2);
}

template<class TDataType, size_t... Dims>
constexpr TTensor<bool, Dims...> operator<(
  const TTensor<TDataType, Dims...>& t,
  TDataType val) {

  return ApplyFunction<sizeof...(Dims)>([val](TDataType v) {
    return v < val;
  }, t);
}

template<class TDataType, size_t... Dims>
constexpr TTensor<bool, Dims...> operator>(
  const TTensor<TDataType, Dims...>& t1,
  const TTensor<TDataType, Dims...>& t2) {

  return ApplyFunction<sizeof...(Dims)>([](TDataType v1, TDataType v2) {
    return v1 > v2;
  }, t1, t2);
}

template<class TDataType, size_t... Dims>
constexpr TTensor<bool, Dims...> operator>(
  const TTensor<TDataType, Dims...>& t,
  TDataType val) {

  return ApplyFunction<sizeof...(Dims)>([val](TDataType v) {
    return v > val;
  }, t);
}

template<class TDataType, size_t... Dims>
constexpr TTensor<bool, Dims...> operator<=(
  const TTensor<TDataType, Dims...>& t1,
  const TTensor<TDataType, Dims...>& t2) {

  return ApplyFunction<sizeof...(Dims)>([](TDataType v1, TDataType v2) {
    return v1 <= v2;
  }, t1, t2);
}

template<class TDataType, size_t... Dims>
constexpr TTensor<bool, Dims...> operator<=(
  const TTensor<TDataType, Dims...>& t,
  TDataType val) {

  return ApplyFunction<sizeof...(Dims)>([val](TDataType v) {
    return v <= val;
  }, t);
}

template<class TDataType, size_t... Dims>
constexpr TTensor<bool, Dims...> operator>=(
  const TTensor<TDataType, Dims...>& t1,
  const TTensor<TDataType, Dims...>& t2) {

  return ApplyFunction<sizeof...(Dims)>([](TDataType v1, TDataType v2) {
    return v1 >= v2;
  }, t1, t2);
}

template<class TDataType, size_t... Dims>
constexpr TTensor<bool, Dims...> operator>=(
  const TTensor<TDataType, Dims...>& t,
  TDataType val) {

  return ApplyFunction<sizeof...(Dims)>([val](TDataType v) {
    return v >= val;
  }, t);
}

template<size_t... Dims>
constexpr TTensor<bool, Dims...> operator!(const TTensor<bool, Dims...>& t) {
  return ApplyFunction<sizeof...(Dims)>([](bool v) {
    return !v;
  }, t);
}

template<size_t... Dims>
constexpr TTensor<bool, Dims...> operator&&(
  const TTensor<bool, Dims...>& t1,
  const TTensor<bool, Dims...>& t2) {

  return ApplyFunction<sizeof...(Dims)>([](bool a, bool b) {
    return a && b;
  }, t1, t2);
}

template<size_t... Dims>
constexpr TTensor<bool, Dims...> operator||(
  const TTensor<bool, Dims...>& t1,
  const TTensor<bool, Dims...>& t2) {

  return ApplyFunction<sizeof...(Dims)>([](bool a, bool b) {
    return a || b;
  }, t1, t2);
}

template<size_t... Dims>
constexpr bool AllOf(const TTensor<bool, Dims...>& t) {
  if constexpr (sizeof...(Dims) == 0) {
    return t;
  } else {
    for (auto& line : t) {
      if (!AllOf(line)) {
        return false;
      }
    }
    return true;
  }
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
  ApplyFunctionInplace<T::DimensionCount>(static_cast<typename T::TData (*)(typename T::TData)>(std::sqrt), inp);
  return inp;
}

template<CTensor T>
T Log(T inp) {
  ApplyFunctionInplace<T::DimensionCount>(static_cast<typename T::TData (*)(typename T::TData)>(std::log), inp);
  return inp;
}

template<CTensor T>
T Abs(T inp) {
  ApplyFunctionInplace<T::DimensionCount>(static_cast<typename T::TData (*)(typename T::TData)>(std::abs), inp);
  return inp;
}

template<CTensor T>
bool AllClose(const T& t1, const T& t2, typename T::TData eps = 1e-6) {
  return AllOf(Abs(t1 - t2) <= eps);
}

template<size_t Dim, CTensor TResult, CTensor T1, CTensor T2>
void StackAlongTo(TResult& result, const T1& a, const T2& b) {
  if constexpr (Dim == 0) {
    for (size_t i = 0; i < a.Size(); ++i) {
      result[i] = a[i];
    }
    for (size_t i = 0; i < b.Size(); ++i) {
      result[i + a.Size()] = b[i];
    }
  } else {
    for (size_t i = 0; i < result.Size(); ++i) {
      StackAlongTo<Dim - 1>(result[i], a[i], b[i]);
    }
  }
}

template<size_t Dim, CTensor T1, CTensor T2>
auto StackAlong(const T1& a, const T2& b) {
  helpers::TStackAlongResult<Dim, T1, T2> result;
  StackAlongTo<Dim>(result, a, b);
  return result;
}

template<size_t Dim, size_t Size, CTensor TRet1, CTensor TRet2, CTensor TSource>
void SplitAlongTo(TRet1& a, TRet2& b, const TSource& source) {
  if constexpr (Dim == 0) {
    for (size_t i = 0; i < Size; ++i) {
      a[i] = source[i];
    }
    for (size_t i = Size; i < source.Size(); ++i) {
      b[i - Size] = source[i];
    }
  } else {
    for (size_t i = 0; i < source.Size(); ++i) {
      SplitAlongTo<Dim - 1, Size>(a[i], b[i], source[i]);
    }
  }
}

template<size_t Dim, size_t Size, CTensor T>
auto SplitAlong(const T& t) {
  helpers::TSplitAlongResult<Dim, Size, T> result;
  SplitAlongTo<Dim, Size>(result.first, result.second, t);
  return result;
}

template<CTensor T>
constexpr typename T::TData Sum(const T& arg) {
  if constexpr (T::DimensionCount == 0) {
    return arg;
  } else {
    typename T::TData sm = 0;
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
