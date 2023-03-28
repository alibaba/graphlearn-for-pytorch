/* Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef GRAPHLEARN_TORCH_INCLUDE_FUNC_FACTORY_H_
#define GRAPHLEARN_TORCH_INCLUDE_FUNC_FACTORY_H_

#include <functional>
#include <unordered_map>

namespace graphlearn_torch {

/// A factory for function call with the same return type and argument types.
template <typename KeyType, typename FuncReturnType, typename... FuncArgs>
class FunctionFactory {
public:
  typedef std::function<FuncReturnType(FuncArgs...)> Function;
  typedef std::unordered_map<KeyType, Function> FactoryMap;

  static FunctionFactory& Get() {
    static FunctionFactory instance;
    return instance;
  }

  FunctionFactory(const FunctionFactory&) = delete;
  FunctionFactory& operator=(const FunctionFactory&) = delete;
  FunctionFactory(FunctionFactory&&) = delete;
  FunctionFactory& operator=(FunctionFactory&&) = delete;

  bool Register(const KeyType& key, Function func) {
    if (factory_map_.find(key) != factory_map_.end()) {
      return false;
    }
    factory_map_[key] = std::move(func);
    return true;
  }

  template <
    typename _RT = FuncReturnType,
    std::enable_if_t<!std::is_same<_RT, void>::value, int> = 0
  >
  auto Dispatch(const KeyType& key, FuncArgs... func_args) {
    if (factory_map_.find(key) == factory_map_.end()) {
      return _RT{};
    }
    return factory_map_[key](std::forward<FuncArgs>(func_args)...);
  }

  template <
    typename _RT = FuncReturnType,
    std::enable_if_t<std::is_same<_RT, void>::value, int> = 0
  >
  void Dispatch(const KeyType& key, FuncArgs... func_args) {
    if (factory_map_.find(key) == factory_map_.end()) {
      return;
    }
    factory_map_[key](std::forward<FuncArgs>(func_args)...);
  }

private:
  FactoryMap factory_map_;

  FunctionFactory() = default;
};

}  // namespace graphlearn_torch

#endif // GRAPHLEARN_TORCH_INCLUDE_FUNC_FACTORY_H_
