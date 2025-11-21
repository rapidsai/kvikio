/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <memory>

namespace kvikio::detail {
/**
 * @brief Type-erased function wrapper that can hold either a copyable or move-only callable. This
 * class avoids the limitation and inconvenience of std::function whose target has to be copyable.
 *
 * @todo Use small object optimization to avoid heap allocation.
 * @note This class will be superseded in the future by C++23 std::move_only_function.
 */
class FunctionWrapper {
 private:
  struct InnerBase {
    virtual void operator()() = 0;

    virtual ~InnerBase() = default;
  };

  template <typename F>
  struct Inner : InnerBase {
    using F_decay = std::decay_t<F>;
    static_assert(std::is_invocable_r_v<void, F_decay>);

    explicit Inner(F&& f) : _f(std::forward<F>(f)) {}

    void operator()() override { std::invoke(_f); }

    ~Inner() override = default;

    F_decay _f;
  };

  std::unique_ptr<InnerBase> _callable;

 public:
  /**
   * @brief Constructor. Create a function wrapper that can hold either a copyable or move-only
   * callable.
   *
   * @tparam F Callable type.
   * @param f Callable.
   */
  template <typename F>
  FunctionWrapper(F&& f) : _callable(std::make_unique<Inner<F>>(std::forward<F>(f)))
  {
  }

  FunctionWrapper() = default;

  FunctionWrapper(FunctionWrapper&&) noexcept            = default;
  FunctionWrapper& operator=(FunctionWrapper&&) noexcept = default;

  FunctionWrapper(const FunctionWrapper&)            = delete;
  FunctionWrapper& operator=(const FunctionWrapper&) = delete;

  void operator()()
  {
    if (!_callable) { throw std::bad_function_call(); }
    _callable->operator()();
  }

  /**
   * @brief Conversion function that tells whether the wrapper has a target (true) or is empty
   * (false).
   *
   * @return Boolean answer.
   */
  operator bool() const noexcept { return _callable != nullptr; }
};

}  // namespace kvikio::detail
