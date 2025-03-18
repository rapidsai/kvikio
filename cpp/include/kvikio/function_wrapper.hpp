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

#include <functional>
#include <memory>

namespace kvikio {
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

  operator bool() { return _callable != nullptr; }
};

using FunctionWrapper = SimpleFunctionWrapper;
}  // namespace kvikio
