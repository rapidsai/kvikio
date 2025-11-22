/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <memory>

namespace kvikio::detail {
/**
 * @brief Type-erased function wrapper that can hold a copyable or move-only callable with signature
 * void(). Unlike std::function, this wrapper is move-only and cannot be copied.
 *
 * @todo Use small buffer optimization to avoid heap allocation for small callables.
 * @note This class will be superseded by C++23's std::move_only_function.
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
   * @brief Construct a function wrapper from a callable object. The callable must be invocable with
   * no arguments and return void. It can be either copyable or move-only (e.g., a lambda capturing
   * std::unique_ptr).
   *
   * @tparam F Callable type.
   * @param f Callable object to wrap. Will be moved or copied into the wrapper.
   */
  template <typename F>
  FunctionWrapper(F&& f) : _callable(std::make_unique<Inner<F>>(std::forward<F>(f)))
  {
  }

  /**
   * @brief Default constructor. Creates an empty wrapper with no callable target.
   */
  FunctionWrapper() = default;

  FunctionWrapper(FunctionWrapper&&) noexcept            = default;
  FunctionWrapper& operator=(FunctionWrapper&&) noexcept = default;

  FunctionWrapper(const FunctionWrapper&)            = delete;
  FunctionWrapper& operator=(const FunctionWrapper&) = delete;

  /**
   * @brief Invoke the wrapped callable.
   *
   * @exception std::bad_function_call if the wrapper is empty (default-constructed or moved-from).
   */
  void operator()()
  {
    if (!_callable) { throw std::bad_function_call(); }
    _callable->operator()();
  }

  /**
   * @brief Check whether the wrapper contains a callable target.
   *
   * @return true if the wrapper contains a callable, false if it is empty.
   */
  explicit operator bool() const noexcept { return _callable != nullptr; }

  /**
   * @brief Reset the wrapper to an empty state, destroying the contained callable if any.
   */
  void reset() noexcept { _callable.reset(); }
};

}  // namespace kvikio::detail
