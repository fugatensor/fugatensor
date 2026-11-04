#pragma once

#include <nanobind/nanobind.h>
#include <cstdint>
#include <memory>
#include <vector>

namespace fuga::lazy {
namespace nb = nanobind;
class RuntimeArena;

struct TensorHandle {
  std::shared_ptr<RuntimeArena> arena;
  std::uint32_t value = 0;
  nb::object engine;

  TensorHandle(std::shared_ptr<RuntimeArena>, std::uint32_t, nb::object);
  TensorHandle(const TensorHandle &);
  TensorHandle(TensorHandle &&) noexcept;
  ~TensorHandle();
  nb::tuple shape() const;
  nb::object dtype() const;
  nb::object storage() const;
  std::uint32_t typeId() const;
  const std::vector<std::int64_t> &dimensions() const;
};

nb::tuple bindPending(nb::handle engine, nb::handle tensorType, nb::handle metadata,
                      nb::tuple inputs, nb::tuple dependencies);
} // namespace fuga::lazy
