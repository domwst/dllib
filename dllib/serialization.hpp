#pragma once

#include <ostream>
#include <istream>
#include <dllib/tensor.hpp>

namespace dllib {

namespace helpers {

template<class T>
concept HasDumpMember = requires(const T& value, std::ostream& out) {
  value.Dump(out);
};

template<class T>
concept HasDumpPointerMember = requires(const T& value, std::ostream& out) {
  value->Dump(out);
};

template<class T>
concept HasLoadMember = requires(T& value, std::istream& in) {
  value.Load(in);
};

template<class T>
concept HasLoadPointerMember = requires(T& value, std::istream& in) {
  value->Load(in);
};

template<class T>
concept HasSerializationFields = requires(T& value) {
  value.GetSerializationFields();
};

} //  namespace helpers

template<helpers::HasDumpMember T>
void Dump(std::ostream& out, const T& obj) {
  obj.Dump(out);
}

template<helpers::HasDumpPointerMember T>
void Dump(std::ostream& out, const T& obj) {
  obj->Dump(out);
}

template<class T>
std::enable_if_t<std::is_integral_v<T> || std::is_floating_point_v<T>> Dump(std::ostream& out, const T& obj) {
  out.write(reinterpret_cast<const char*>(&obj), sizeof(obj));
}

template<class T, size_t... Dims>
void Dump(std::ostream& out, const TTensor<T, Dims...>& tensor) {
  if constexpr (sizeof...(Dims) == 0) {
    Dump(out, tensor.Data());
  } else {
    for (auto& x : tensor) {
      Dump(out, x);
    }
  }
}

template<helpers::HasSerializationFields T>
void Dump(std::ostream& out, const T& obj) {
  auto fields = obj.GetSerializationFields();
  constexpr size_t n = std::tuple_size_v<decltype(fields)>;
  [&out, &fields]<size_t... i>(std::index_sequence<i...>) {
    (Dump(out, std::get<i>(fields)),...);
  }(std::make_index_sequence<n>{});
}

template<helpers::HasLoadMember T>
void Load(std::istream& in, T& obj) {
  obj.Load(in);
}

template<helpers::HasLoadPointerMember T>
void Load(std::istream& in, T& obj) {
  obj->Load(in);
}

template<class T>
std::enable_if_t<std::is_integral_v<T> || std::is_floating_point_v<T>> Load(std::istream& in, T& obj) {
  assert(in.read(reinterpret_cast<char*>(&obj), sizeof(obj)));
}

template<class T, size_t... Dims>
void Load(std::istream& in, TTensor<T, Dims...>& tensor) {
  if constexpr (sizeof...(Dims) == 0) {
    Load(in, tensor.Data());
  } else {
    for (auto& x : tensor) {
      Load(in, x);
    }
  }
}

template<helpers::HasSerializationFields T>
void Load(std::istream& in, T& obj) {
  auto fields = obj.GetSerializationFields();
  constexpr size_t n = std::tuple_size_v<decltype(fields)>;
  [&in, &fields]<size_t... i>(std::index_sequence<i...>) {
    (Load(in, std::get<i>(fields)),...);
  }(std::make_index_sequence<n>{});
}

}  // namespace dllib
