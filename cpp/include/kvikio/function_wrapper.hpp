/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions andc
 * limitations under the License.
 */

#pragma once

#include <functional>
#include <memory>

namespace kvikio {
/**
 * @brief Type-erased function wrapper that can hold either a copyable or move-only callable. This
 * class avoids the limitation and inconvenience of std::function whose target has to be copyable.
 *
 * @todo Use small object optimization to avoid heap allocation.
 * @note This class will be deprecated in the far future when C++23 is adopted that offers
 * std::move_only_function.
 */
class SimpleFunctionWrapper {
 private:
  struct inner_base {
    virtual void operator()() = 0;

    virtual ~inner_base() = default;
  };

  template <typename F>
  struct inner : inner_base {
    using F_decay = std::decay_t<F>;
    static_assert(std::is_invocable_r_v<void, F_decay>);

    inner(F&& f) : _f(std::forward<F>(f)) {}

    void operator()() override { std::invoke(_f); }

    ~inner() override = default;

    F_decay _f;
  };

  std::unique_ptr<inner_base> _callable;

 public:
  /**
   * @brief Constructor. Create a function wrapper that can hold either a copyable or move-only
   * callable.
   *
   * @tparam F Callable type.
   * @param f Callable.
   */
  template <typename F>
  SimpleFunctionWrapper(F&& f) : _callable(std::make_unique<inner<F>>(std::forward<F>(f)))
  {
    using F_decay = std::decay_t<F>;
    static_assert(std::is_invocable_r_v<void, F_decay>);
  }

  SimpleFunctionWrapper() = default;

  SimpleFunctionWrapper(SimpleFunctionWrapper&&)            = default;
  SimpleFunctionWrapper& operator=(SimpleFunctionWrapper&&) = default;

  SimpleFunctionWrapper(const SimpleFunctionWrapper&)            = delete;
  SimpleFunctionWrapper& operator=(const SimpleFunctionWrapper&) = delete;

  void operator()() { return _callable->operator()(); }

  /**
   * @brief Conversion function that tells whether the wrapper has a target (true) or is empty
   * (false).
   *
   * @return Boolean answer.
   */
  operator bool() { return _callable != nullptr; }
};

using FunctionWrapper = SimpleFunctionWrapper;
}  // namespace kvikio
