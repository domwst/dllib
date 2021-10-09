#pragma once

#include <array>
#include <cassert>
#include <type_traits>
#include <utility>

namespace dllib {

template<class TData, std::size_t... Dims>
class TTensor;

namespace helpers {
    template<class>
    struct TIsTensorHelper : std::false_type {};

    template<class TData, std::size_t... Dims>
    struct TIsTensorHelper<TTensor<TData, Dims...>> : std::true_type {};

    template<class, class>
    struct TIsTensorOfTypeHelper : std::false_type {};

    template<class TData, std::size_t... Dims>
    struct TIsTensorOfTypeHelper<TTensor<TData, Dims...>, TData> : std::true_type {};

    template<class, std::size_t...>
    struct TIsTensorWithDimsHelper : std::false_type {};

    template<class TData, std::size_t... Dims>
    struct TIsTensorWithDimsHelper<TTensor<TData, Dims...>, Dims...> : std::true_type {};
}


template<class T>
constexpr bool VIsTensor = helpers::TIsTensorHelper<std::remove_cvref_t<T>>::value;

template<class T>
struct TIsTensor : std::bool_constant<VIsTensor<T>> {};

template<class T>
concept CTensor = VIsTensor<T>;


template<class T, class TData>
constexpr bool VIsTensorOfType = helpers::TIsTensorOfTypeHelper<std::remove_cvref_t<T>, TData>::value;

template<class T, class TData>
struct TIsTensorOfType : std::bool_constant<VIsTensorOfType<T, TData>> {};

template<class T, class TData>
concept CTensorOfType = VIsTensorOfType<T, TData>;


template<class T, std::size_t... Dims>
constexpr bool VIsTensorWithDims = helpers::TIsTensorWithDimsHelper<std::remove_cvref_t<T>, Dims...>::value;

template<class T, std::size_t... Dims>
struct TIsTensorWithDims : std::bool_constant<VIsTensorWithDims<T, Dims...>> {};

template<class T, std::size_t... Dims>
concept CTensorWithDims = VIsTensorWithDims<T, Dims...>;


namespace helpers {
    template<class Lambda, int=(Lambda{}(), 0)>
    constexpr bool IsConstexpr(Lambda) {
        return true;
    }

    constexpr bool IsConstexpr(...) {
        return false;
    }

    template<class T>
    concept CHasBeginEnd = requires(const T& value) {
        { std::begin(value) };
        { std::end(value) };
    };

    template<std::size_t N>
    constexpr std::enable_if_t<(N > 0), std::array<std::size_t, N>>
    ChangeLastElement(std::array<std::size_t, N> Arr, std::size_t NewVal) {
        Arr.back() = NewVal;
        return Arr;
    }


    template<class, std::array, class>
    struct MakeTensorHelper;

    template<class TData, std::size_t N, std::array<std::size_t, N> Dims, std::size_t... I>
    struct MakeTensorHelper<TData, Dims, std::index_sequence<I...>> {
        using type = TTensor<TData, Dims[I]...>;
    };

    template<class, std::array>
    struct MakeTensor;

    template<class TData, std::size_t N, std::array<std::size_t, N> Dims>
    struct MakeTensor<TData, Dims> {
        using type = typename MakeTensorHelper<TData, Dims, std::make_index_sequence<N>>::type;
    };

    template<class TData, std::array Dims>
    using TMakeTensor = typename MakeTensor<TData, Dims>::type;

    template<class, std::size_t>
    struct ReplaceLastDimension;

    template<class TData, std::size_t... Dims, std::size_t NewLastDim>
    struct ReplaceLastDimension<TTensor<TData, Dims...>, NewLastDim> {
        using type = TMakeTensor<TData, ChangeLastElement(std::array{Dims...}, NewLastDim)>;
    };

    template<class T, std::size_t LastDim>
    using TReplaceLastDimension = typename ReplaceLastDimension<T, LastDim>::type;
}

template<class TData>
class TTensor<TData> {
public:

    using DataType = TData;
    static constexpr std::size_t TotalElements = 1;
    static constexpr std::size_t DimensionCount = 0;

    template<std::size_t N>
    using SubTensor = std::conditional_t<N == 0, TTensor, void>;

    constexpr TTensor(TData val = 0) {
        FillWith(val);
    }

    constexpr TTensor& operator=(TData val) {
        data_ = val;
        return *this;
    }

    template<std::size_t... NewDims>
    const TTensor<TData, NewDims...>& view() const {
        static_assert(TTensor<TData, NewDims...>::TotalElements == TotalElements);
        return *reinterpret_cast<const TTensor<TData, NewDims...>*>(this);
    }

    template<std::size_t... NewDims>
    TTensor<TData, NewDims...>& view() {
        static_assert(TTensor<TData, NewDims...>::TotalElements == TotalElements);
        return *reinterpret_cast<TTensor<TData, NewDims...>*>(this);
    }

    constexpr TTensor& FillWith(TData val) {
        return (*this) = val;
    }

    constexpr TData data() const {
        return data_;
    }

    constexpr TData& data() {
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

template<class TData, std::size_t FirstDim, std::size_t... OtherDims>
class TTensor<TData, FirstDim, OtherDims...> {
public:

    using DataType = TData;
    using ElementType = TTensor<TData, OtherDims...>;
    using ContainerType = std::array<ElementType, FirstDim>;

    static constexpr std::size_t TotalElements = ElementType::TotalElements * FirstDim;
    static constexpr std::size_t DimensionCount = sizeof...(OtherDims) + 1;

    template<std::size_t N>
    using SubTensor = std::conditional_t<N == DimensionCount, TTensor, typename ElementType::template SubTensor<N>>;

    constexpr TTensor(TData val = 0) {
        FillWith(val);
    }

    template<class TForwardIt>
    constexpr TTensor(TForwardIt begin, TForwardIt end) {
        if constexpr (helpers::IsConstexpr([&begin, &end]() { return std::distance(begin, end); })) {
            static_assert(std::distance(begin, end) == FirstDim);
        } else {
            assert(std::distance(begin, end) == FirstDim);
        }
        for (std::size_t i = 0; i < FirstDim; ++i) {
            data_[i] = *begin;
            ++begin;
        }
    }

    template<helpers::CHasBeginEnd T>
    constexpr TTensor(const T& value) : TTensor(std::begin(value), std::end(value)) {}

    constexpr TTensor& operator+=(const TTensor& other) {
        for (std::size_t i = 0; i < FirstDim; ++i) {
            data_[i] += other[i];
        }
        return *this;
    }

    constexpr TTensor& operator+=(TData val) {
        for (std::size_t i = 0; i < FirstDim; ++i) {
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
        for (std::size_t i = 0; i < FirstDim; ++i) {
            data_[i] -= other[i];
        }
        return *this;
    }

    constexpr TTensor& operator-=(TData val) {
        for (std::size_t i = 0; i < FirstDim; ++i) {
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
        for (std::size_t i = 0; i < FirstDim; ++i) {
            data_[i] *= other[i];
        }
        return *this;
    }

    constexpr TTensor& operator*=(TData val) {
        for (std::size_t i = 0; i < FirstDim; ++i) {
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
        for (std::size_t i = 0; i < FirstDim; ++i) {
            data_[i] /= other[i];
        }
        return *this;
    }

    constexpr TTensor& operator/=(TData val) {
        for (std::size_t i = 0; i < FirstDim; ++i) {
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

    constexpr const ElementType& operator[](std::size_t idx) const {
        return data_[idx];
    }

    constexpr ElementType& operator[](std::size_t idx) {
        return data_[idx];
    }

    template<std::size_t... NewDims>
    constexpr const TTensor<TData, NewDims...>& view() const {
        static_assert(TTensor<TData, NewDims...>::TotalElements == TotalElements);
        return *reinterpret_cast<const TTensor<TData, NewDims...>*>(this);
    }

    template<std::size_t... NewDims>
    constexpr TTensor<TData, NewDims...>& view() {
        static_assert(TTensor<TData, NewDims...>::TotalElements == TotalElements);
        return *reinterpret_cast<TTensor<TData, NewDims...>*>(this);
    }

    constexpr TTensor& FillWith(TData val) {
        for (std::size_t i = 0; i < FirstDim; ++i) {
            data_[i].FillWith(val);
        }
        return *this;
    }

    constexpr const ContainerType& data() const {
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

    constexpr static std::size_t size() {
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

template<std::size_t DimToStop, class TFunction, CTensor Tensor>
constexpr std::enable_if_t<(Tensor::DimensionCount >= DimToStop), void>
ApplyFunctionInplace(TFunction&& function, Tensor& tensor) {
    if constexpr (Tensor::DimensionCount == DimToStop) {
        tensor = function(tensor.data());
    } else {
        for (std::size_t i = 0; i < tensor.size(); ++i) {
            ApplyFunctionInplace<DimToStop>(function, tensor[i]);
        }
    }
}

template<std::size_t DimToStop, class TFunction, class TSourceData, std::size_t... Dims, CTensorWithDims<Dims...> TensorResult>
constexpr std::enable_if_t<(sizeof...(Dims) >= DimToStop), void>
ApplyFunction(TFunction&& function, const TTensor<TSourceData, Dims...>& source, TensorResult& result) {
    if constexpr (sizeof...(Dims) == DimToStop) {
        result = function(source.data());
    } else {
        for (std::size_t i = 0; i < source.size(); ++i) {
            ApplyFunction<DimToStop>(function, source[i], result[i]);
        }
    }
}

template<class TRetData, std::size_t DimToStop, class TFunction, class TArgumentData, std::size_t... Dims>
constexpr TTensor<TRetData, Dims...> ApplyFunction(TFunction&& function, const TTensor<TArgumentData, Dims...>& arg) {
    TTensor<TRetData, Dims...> result;
    ApplyFunction<DimToStop>(std::move(function), arg, result);
    return result;
}

template<std::size_t DimToStop, class TFunction, CTensor ArgumentTensor>
constexpr auto ApplyFunction(TFunction&& function, const ArgumentTensor& arg) {
    if constexpr (DimToStop == 0) {
        return ApplyFunction<
            std::invoke_result_t<
                TFunction,
                typename ArgumentTensor::template SubTensor<DimToStop>
            >,
            DimToStop
        >(std::move(function), arg);
    } else {
        return ApplyFunction<
            typename std::invoke_result_t<
                TFunction,
                typename ArgumentTensor::template SubTensor<DimToStop>
            >::DataType,
            DimToStop
        >(std::move(function), arg);
    }
}

template<class TData, std::size_t Dim1, std::size_t Dim2, std::size_t Dim3>
constexpr void MatrixMultiplication(
    const TTensor<TData, Dim1, Dim2>& matrix1,
    const TTensor<TData, Dim2, Dim3>& matrix2,
    TTensor<TData, Dim1, Dim3>& result) {

    result.FillWith(0);
    for (std::size_t i = 0; i < Dim1; ++i) {
        for (std::size_t j = 0; j < Dim2; ++j) {
            for (std::size_t k = 0; k < Dim3; ++k) {
                result[i][k] += matrix1[i][j] * matrix2[j][k];
            }
        }
    }
}

template<class TData, std::size_t Dim1, std::size_t Dim2, std::size_t Dim3>
constexpr TTensor<TData, Dim1, Dim3> MatrixMultiplication(
    const TTensor<TData, Dim1, Dim2>& matrix1,
    const TTensor<TData, Dim2, Dim3>& matrix2) {

    TTensor<TData, Dim1, Dim3> result;
    MatrixMultiplication(matrix1, matrix2, result);
    return result;
}

template<class TData, std::size_t FromDim, std::size_t ToDim, CTensorOfType<TData> Tensor>
constexpr std::enable_if_t<(Tensor::DimensionCount > 2), void> MatrixMultiplication(
    const Tensor& tensor,
    const TTensor<TData, FromDim, ToDim>& matrix,
    helpers::TReplaceLastDimension<Tensor, ToDim>& result) {

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        MatrixMultiplication(tensor[i], matrix, result[i]);
    }
}

template<class TData, std::size_t FromDim, std::size_t ToDim, CTensorOfType<TData> Tensor>
constexpr std::enable_if_t<
    (Tensor::DimensionCount > 2),
    helpers::TReplaceLastDimension<Tensor, ToDim>
> MatrixMultiplication(const Tensor& tensor, const TTensor<TData, FromDim, ToDim>& matrix) {

    helpers::TReplaceLastDimension<Tensor, ToDim> result;
    MatrixMultiplication(tensor, matrix, result);
    return result;
}

template<class TData, std::size_t FromDim, std::size_t ToDim, CTensorOfType<TData> Tensor>
constexpr std::enable_if_t<(Tensor::DimensionCount > 2), void>
MatrixMultiplication(
    const TTensor<TData, ToDim, FromDim>& matrix,
    const Tensor& tensor,
    helpers::TReplaceLastDimension<Tensor, ToDim>& result) {

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        MatrixMultiplication(matrix, tensor[i], result[i]);
    }
}

template<class TData, std::size_t FromDim, std::size_t ToDim, CTensorOfType<TData> Tensor>
constexpr std::enable_if_t<
    (Tensor::DimensionCount > 2),
    helpers::TReplaceLastDimension<Tensor, ToDim>
> MatrixMultiplication(const TTensor<TData, ToDim, FromDim>& matrix, const Tensor& tensor) {

    helpers::TReplaceLastDimension<Tensor, ToDim> result;
    MatrixMultiplication(matrix, tensor, result);
    return result;
}

}