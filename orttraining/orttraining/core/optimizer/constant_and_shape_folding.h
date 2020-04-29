// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@class ConstantAndShapeFolding

Transformer that traverses the graph top-down, run shape inference and performs constant folding alternately,
i.e., it statically computes parts of the graph that rely only on constant initializers and Shape nodes with known input sizes.
*/
class ConstantAndShapeFolding : public GraphTransformer {
 public:
  /** Constant folding will not be applied to nodes that have one of initializers from excluded_initializers as input.
      For pre-training, the trainable weights are those initializers to be excluded. */
  ConstantAndShapeFolding(const std::unordered_set<std::string>& compatible_execution_providers = {},
                          const std::unordered_set<std::string>& excluded_initializers = {}) noexcept
      : GraphTransformer("ConstantAndShapeFolding", compatible_execution_providers), excluded_initializers_(excluded_initializers) {}

 private:
  /** Constant folding will not be applied to nodes whose op_type is included in this set.
      All non-deterministic operators should be included in this set. */
  const std::unordered_set<std::string> excluded_op_types_ =
      {"RandomUniform", "RandomNormal", "RandomUniformLike", "RandomNormalLike", "Multinomial"};

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  const std::unordered_set<std::string> excluded_initializers_;
};

}  // namespace onnxruntime
