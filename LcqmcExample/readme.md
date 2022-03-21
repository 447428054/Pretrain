# 任务描述

根据选取数据集，转为预训练格式数据，完成roberta、albert的预训练，并对比在该数据集上，预训练前后的具体任务指标。

# 任务数据集

LCQMC数据集

| 训练集 | 验证集 | 测试集 |
| :----: | :----: | :----: |
|    238766    |    8802    |     12500   |

LCQMC数据集的长度分布如下：

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h0gke2qbgrj20hs0dcglw.jpg)

# 实验设置

代码链接：https://github.com/447428054/Pretrain/tree/master

预训练环境：tensorflow1.14

预训练代码执行顺序：

1. bash create_pretrain_data_lz.sh
2. bash pretrain_lz.sh

LCQMC微调代码:

1. python task_sentence_similarity_lcqmc_roberta.py
2. python task_sentence_similarity_lcqmc_albert.py

TIPS:

记得修改文件路径

# 预训练数据生成

预训练代码读取生成的record，数据处理代码首先读取不同文件，每个文件格式为：**每一行存放一个句子，不同文档之间以空行分割**

我们将LCQMC中相似的句子作为一个文档，不相似的分开

```
谁有狂三这张高清的

这张高清图，谁有

英雄联盟什么英雄最好
英雄联盟最好英雄是什么

这是什么意思，被蹭网吗

```

## roberta的预训练数据处理

1. 每个文件中，一个sentence占一行，不同document之间加一个空行分割
[['有', '人', '知', '道', '叫', '什', '么', '名', '字', '吗', '[UNK]', '？'], ['有', '人', '知', '道', '名', '字', '吗']]

2. 从一个文档中连续的获得文本，直到达到最大长度。如果是从下一个文档中获得，那么加上一个分隔符.将长度限制修改了，因为lcqmc句子都偏短
['有', '人', '知', '道', '叫', '什', '么', '名', '字', '吗', '[UNK]', '？', '有', '人', '知', '道', '名', '字', '吗']

3. 对于获取之后的文本，进行全词分词： 判断每个字符起始长度3以内的，是否在分词里面，在的话添加##标记
['有', '##人', '知', '##道', '叫', '什', '##么', '名', '##字', '吗', '[UNK]', '？', '有', '##人', '知', '##道', '名', '##字', '吗']

4. 对获得的token序列，进行掩码:返回 掩码结果，掩码的位置，掩码的标签
['[CLS]', '有', '人', '知', '道', '叫', '什', '么', '名', '字', '吗', '[UNK]', '[MASK]', '有', '人', '[MASK]', '[MASK]', '名', '字', '吗', '[SEP]']
[12, 15, 16]
['？', '知', '##道']


run_pretraining.py需要注释TPU的引用
```
# *tpu_cluster_resolver* = tf.contrib.cluster_resolver.TPUClusterResolver( # TODO
#       tpu=FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
```


## albert的预训练数据处理

1. 每个文件中，一个sentence占一行，不同document之间加一个空行分割
[['有', '人', '知', '道', '叫', '什', '么', '名', '字', '吗', '[UNK]', '？'], ['有', '人', '知', '道', '名', '字', '吗']]

2. 从一个文档中获取sentence,sentece进行全词分词，当长度达到最大长度或者遍历完整个文档了，A[SEP]B 随机分割句子，50%概率交换顺序，得到SOP标签
tokenA:['有', '##人', '知', '##道', '叫', '什', '##么', '名', '##字', '吗', '[UNK]', '？']
tokenB:['有', '##人', '知', '##道', '名', '##字', '吗']

只有一句话,构不成SOP任务的就continue

3. 对获得的token序列，进行掩码:返回 掩码结果，掩码的位置，掩码的标签
tokens:['[CLS]', '有', '人', '知', '道', '叫', '什', '么', '名', '[MASK]', '吗', '[UNK]', '？', '[SEP]', '[MASK]', '人', '知', '[MASK]', '名', '字', '吗', '[SEP]']
masked_lm_positions:[9, 14, 17]
masked_lm_labels:['##字', '有', '##道']
is_random_next:False



# 预训练代码

## 模型结构

### Roberta

**整个模型结构与BERT相同，整体流程如下：**

1. **输入的token 经过embedding**
2. **再加上token type id 与position embedding**
3. **进入transfomer层，每一个transformer又由多头attention、层归一化、残差结构、前馈神经网络构成**
4. **获取CLS输出与整个句子的输出**

**整个代码结构跟流程相同：**

```
    with tf.variable_scope(scope, default_name="bert"):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        (self.embedding_output, self.embedding_table) = embedding_lookup(
            input_ids=input_ids,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=use_one_hot_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = embedding_postprocessor(
            input_tensor=self.embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)

      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        attention_mask = create_attention_mask_from_input_mask(
            input_ids, input_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        self.all_encoder_layers = transformer_model(
            input_tensor=self.embedding_output,
            attention_mask=attention_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True)

      self.sequence_output = self.all_encoder_layers[-1] # [batch_size, seq_length, hidden_size]
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        self.pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range))
```

#### embedding_lookup

**与常规神经网络中词嵌入类似**

```
def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
  """Looks up words embeddings for id tensor.
  Args:
    input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
      ids.
    vocab_size: int. Size of the embedding vocabulary.
    embedding_size: int. Width of the word embeddings.
    initializer_range: float. Embedding initialization range.
    word_embedding_name: string. Name of the embedding table.
    use_one_hot_embeddings: bool. If True, use one-hot method for word
      embeddings. If False, use `tf.gather()`.
  Returns:
    float Tensor of shape [batch_size, seq_length, embedding_size].
  """
  # This function assumes that the input is of shape [batch_size, seq_length,
  # num_inputs].
  #
  # If the input is a 2D tensor of shape [batch_size, seq_length], we
  # reshape to [batch_size, seq_length, 1].
  if input_ids.shape.ndims == 2:
    input_ids = tf.expand_dims(input_ids, axis=[-1])

  embedding_table = tf.get_variable(
      name=word_embedding_name,
      shape=[vocab_size, embedding_size],
      initializer=create_initializer(initializer_range))

  flat_input_ids = tf.reshape(input_ids, [-1])
  if use_one_hot_embeddings:
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
    output = tf.matmul(one_hot_input_ids, embedding_table)
  else:
    output = tf.gather(embedding_table, flat_input_ids)

  input_shape = get_shape_list(input_ids)

  output = tf.reshape(output,
                      input_shape[0:-1] + [input_shape[-1] * embedding_size])
  return (output, embedding_table)
```

#### embedding_postprocessor

**加上token type id与可学习的position id 词嵌入，postprocessor指在embedding之后进行层归一化与dropout**

```
def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
  """Performs various post-processing on a word embedding tensor.
  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    token_type_embedding_name: string. The name of the embedding table variable
      for token type ids.
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.
  Returns:
    float tensor with same shape as `input_tensor`.
  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]

  output = input_tensor

  if use_token_type:
    if token_type_ids is None:
      raise ValueError("`token_type_ids` must be specified if"
                       "`use_token_type` is True.")
    token_type_table = tf.get_variable(
        name=token_type_embedding_name,
        shape=[token_type_vocab_size, width],
        initializer=create_initializer(initializer_range))
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary.
    flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    output += token_type_embeddings

  if use_position_embeddings:
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
      full_position_embeddings = tf.get_variable(
          name=position_embedding_name,
          shape=[max_position_embeddings, width],
          initializer=create_initializer(initializer_range))
      # Since the position embedding table is a learned variable, we create it
      # using a (long) sequence length `max_position_embeddings`. The actual
      # sequence length might be shorter than this, for faster training of
      # tasks that do not have long sequences.
      #
      # So `full_position_embeddings` is effectively an embedding table
      # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
      # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
      # perform a slice.
      position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                     [seq_length, -1])
      num_dims = len(output.shape.as_list())

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size.
      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, width])
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      output += position_embeddings

  output = layer_norm_and_dropout(output, dropout_prob)
  return 
```

#### transformer_model

**对于每一个Transformer结构如下：**

1. **多头attention**
2. **拼接多头输出，经过隐藏层映射**
3. **经过dropout+残差+层归一化**
4. **前馈神经网络**
5. **经过dropout+残差+层归一化**

```
def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".
  This is almost an exact implementation of the original Transformer encoder.
  See the original paper:
  https://arxiv.org/abs/1706.03762
  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
      seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_return_all_layers: Whether to also return all layers or just the final
      layer.
  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.
  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  # We keep the representation as a 2D tensor to avoid re-shaping it back and
  # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
  # help the optimizer.
  prev_output = reshape_to_matrix(input_tensor)

  all_layer_outputs = []
  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope("layer_%d" % layer_idx):
      layer_input = prev_output

      with tf.variable_scope("attention"):
        attention_heads = []
        with tf.variable_scope("self"):
          attention_head = attention_layer(
              from_tensor=layer_input,
              to_tensor=layer_input,
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length)
          attention_heads.append(attention_head)

        attention_output = None
        if len(attention_heads) == 1:
          attention_output = attention_heads[0]
        else:
          # In the case where we have other sequences, we just concatenate
          # them to the self-attention head before the projection.
          attention_output = tf.concat(attention_heads, axis=-1)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.variable_scope("output"):
          attention_output = tf.layers.dense(
              attention_output,
              hidden_size,
              kernel_initializer=create_initializer(initializer_range))
          attention_output = dropout(attention_output, hidden_dropout_prob)
          attention_output = layer_norm(attention_output + layer_input)

      # The activation is only applied to the "intermediate" hidden layer.
      with tf.variable_scope("intermediate"):
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=create_initializer(initializer_range))

      # Down-project back to `hidden_size` then add the residual.
      with tf.variable_scope("output"):
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm(layer_output + attention_output)
        prev_output = layer_output
        all_layer_outputs.append(layer_output)

  if do_return_all_layers:
    final_outputs = []
    for layer_output in all_layer_outputs:
      final_output = reshape_from_matrix(layer_output, input_shape)
      final_outputs.append(final_output)
    return final_outputs
  else:
    final_output = reshape_from_matrix(prev_output, input_shape)
    return final_output
```

**其中，多头attention结构如下：**

1. **针对输入向量与输出向量，生成Q、K、V向量，隐藏层维度为num heads * head size**

   **self-attention中 输入输出来源相同。Q来源于输入向量，V来源于输出向量。“目的在于计算输入向量 对于 不同输出的 权重**”

   **若需要对注意力进行掩码，对得分减去很大的值，最终softmax之后得到的影响就非常小**

2. **针对Q，K 进行缩放点积计算，再经过softmax**

3. **将第二步结果与V相乘得到上下文向量**

```
def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.
  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.
  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].
  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.
  In practice, the multi-headed attention are done with transposes and
  reshapes rather than actual separate tensors.
  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
      * from_seq_length, num_attention_heads * size_per_head]. If False, the
      output will be of shape [batch_size, from_seq_length, num_attention_heads
      * size_per_head].
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.
  Returns:
    float Tensor of shape [batch_size, from_seq_length,
      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
      true, this will be of shape [batch_size * from_seq_length,
      num_attention_heads * size_per_head]).
  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """

  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`

  from_tensor_2d = reshape_to_matrix(from_tensor)
  to_tensor_2d = reshape_to_matrix(to_tensor)

  # `query_layer` = [B*F, N*H]
  query_layer = tf.layers.dense(
      from_tensor_2d,
      num_attention_heads * size_per_head,
      activation=query_act,
      name="query",
      kernel_initializer=create_initializer(initializer_range))

  # `key_layer` = [B*T, N*H]
  key_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=key_act,
      name="key",
      kernel_initializer=create_initializer(initializer_range))

  # `value_layer` = [B*T, N*H]
  value_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=value_act,
      name="value",
      kernel_initializer=create_initializer(initializer_range))

  # `query_layer` = [B, N, F, H]
  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer` = [B, N, T, H]
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  # `attention_scores` = [B, N, F, T]
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

  if attention_mask is not None:
    # `attention_mask` = [B, 1, F, T]
    attention_mask = tf.expand_dims(attention_mask, axis=[1])

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_scores += adder

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_probs = tf.nn.softmax(attention_scores)

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

  # `value_layer` = [B, T, N, H]
  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])

  # `value_layer` = [B, N, T, H]
  value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

  # `context_layer` = [B, N, F, H]
  context_layer = tf.matmul(attention_probs, value_layer)

  # `context_layer` = [B, F, N, H]
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

  if do_return_2d_tensor:
    # `context_layer` = [B*F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
  else:
    # `context_layer` = [B, F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head])

  return context_layer
```

### Albert

**Albert整体结构与BERT相似，改动有三点：**

1. **词嵌入层由Vocab * Hidden 分解为 Vocab * Embedding + Embedding * Hidden**

2. **跨层参数共享，主要是全连接层与注意力层的共享**

   **Tensorflow 中 通过get variable 与 变量域Variable Scope完成参数共享**

3. **段落连续的SOP任务替换原先NSP任务，SOP任务中文档连续语句为正例，调换顺序后为负例**

**代码中同时更新了层归一化的顺序：pre-Layer Normalization can converge fast and better. check paper: ON LAYER NORMALIZATION IN THE TRANSFORMER ARCHITECTURE**

**模型结构改动主要涉及前两点，接下来我们从代码层面来看这些改动：**

#### embedding_lookup_factorized

**主要拆分为两次矩阵运算，embedding size在中间过渡**

```
def embedding_lookup_factorized(input_ids, # Factorized embedding parameterization provide by albert
                     vocab_size,
                     hidden_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
    """Looks up words embeddings for id tensor, but in a factorized style followed by albert. it is used to reduce much percentage of parameters previous exists.
       Check "Factorized embedding parameterization" session in the paper.
     Args:
       input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
         ids.
       vocab_size: int. Size of the embedding vocabulary.
       embedding_size: int. Width of the word embeddings.
       initializer_range: float. Embedding initialization range.
       word_embedding_name: string. Name of the embedding table.
       use_one_hot_embeddings: bool. If True, use one-hot method for word
         embeddings. If False, use `tf.gather()`.
     Returns:
       float Tensor of shape [batch_size, seq_length, embedding_size].
     """
    # This function assumes that the input is of shape [batch_size, seq_length,
    # num_inputs].
    #
    # If the input is a 2D tensor of shape [batch_size, seq_length], we
    # reshape to [batch_size, seq_length, 1].

    # 1.first project one-hot vectors into a lower dimensional embedding space of size E
    print("embedding_lookup_factorized. factorized embedding parameterization is used.")
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])  # shape of input_ids is:[ batch_size, seq_length, 1]

    embedding_table = tf.get_variable(  # [vocab_size, embedding_size]
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=create_initializer(initializer_range))

    flat_input_ids = tf.reshape(input_ids, [-1])  # one rank. shape as (batch_size * sequence_length,)
    if use_one_hot_embeddings:
        one_hot_input_ids = tf.one_hot(flat_input_ids,depth=vocab_size)  # one_hot_input_ids=[batch_size * sequence_length,vocab_size]
        output_middle = tf.matmul(one_hot_input_ids, embedding_table)  # output=[batch_size * sequence_length,embedding_size]
    else:
        output_middle = tf.gather(embedding_table,flat_input_ids)  # [vocab_size, embedding_size]*[batch_size * sequence_length,]--->[batch_size * sequence_length,embedding_size]

    # 2. project vector(output_middle) to the hidden space
    project_variable = tf.get_variable(  # [embedding_size, hidden_size]
        name=word_embedding_name+"_2",
        shape=[embedding_size, hidden_size],
        initializer=create_initializer(initializer_range))
    output = tf.matmul(output_middle, project_variable) # ([batch_size * sequence_length, embedding_size] * [embedding_size, hidden_size])--->[batch_size * sequence_length, hidden_size]
    # reshape back to 3 rank
    input_shape = get_shape_list(input_ids)  # input_shape=[ batch_size, seq_length, 1]
    batch_size, sequene_length, _=input_shape
    output = tf.reshape(output, (batch_size,sequene_length,hidden_size))  # output=[batch_size, sequence_length, hidden_size]
    return (output, embedding_table, project_variable)
```

#### prelln_transformer_model

将Layer Norm放在Attention前面，使训练过程收敛的更快更好。使用Tensorflow的变量域，完成参数共享。

```
def prelln_transformer_model(input_tensor,
						attention_mask=None,
						hidden_size=768,
						num_hidden_layers=12,
						num_attention_heads=12,
						intermediate_size=3072,
						intermediate_act_fn=gelu,
						hidden_dropout_prob=0.1,
						attention_probs_dropout_prob=0.1,
						initializer_range=0.02,
						do_return_all_layers=False,
						shared_type='all', # None,
						adapter_fn=None):
	"""Multi-headed, multi-layer Transformer from "Attention is All You Need".
	This is almost an exact implementation of the original Transformer encoder.
	See the original paper:
	https://arxiv.org/abs/1706.03762
	Also see:
	https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
	Args:
		input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
		attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
			seq_length], with 1 for positions that can be attended to and 0 in
			positions that should not be.
		hidden_size: int. Hidden size of the Transformer.
		num_hidden_layers: int. Number of layers (blocks) in the Transformer.
		num_attention_heads: int. Number of attention heads in the Transformer.
		intermediate_size: int. The size of the "intermediate" (a.k.a., feed
			forward) layer.
		intermediate_act_fn: function. The non-linear activation function to apply
			to the output of the intermediate/feed-forward layer.
		hidden_dropout_prob: float. Dropout probability for the hidden layers.
		attention_probs_dropout_prob: float. Dropout probability of the attention
			probabilities.
		initializer_range: float. Range of the initializer (stddev of truncated
			normal).
		do_return_all_layers: Whether to also return all layers or just the final
			layer.
	Returns:
		float Tensor of shape [batch_size, seq_length, hidden_size], the final
		hidden layer of the Transformer.
	Raises:
		ValueError: A Tensor shape or parameter is invalid.
	"""
	if hidden_size % num_attention_heads != 0:
		raise ValueError(
				"The hidden size (%d) is not a multiple of the number of attention "
				"heads (%d)" % (hidden_size, num_attention_heads))

	attention_head_size = int(hidden_size / num_attention_heads)

	input_shape = bert_utils.get_shape_list(input_tensor, expected_rank=3)
	batch_size = input_shape[0]
	seq_length = input_shape[1]
	input_width = input_shape[2]

	# The Transformer performs sum residuals on all layers so the input needs
	# to be the same as the hidden size.
	if input_width != hidden_size:
		raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
										 (input_width, hidden_size))

	# We keep the representation as a 2D tensor to avoid re-shaping it back and
	# forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
	# the GPU/CPU but may not be free on the TPU, so we want to minimize them to
	# help the optimizer.
	prev_output = bert_utils.reshape_to_matrix(input_tensor)

	all_layer_outputs = []

	def layer_scope(idx, shared_type):
		if shared_type == 'all':
			tmp = {
				"layer":"layer_shared",
				'attention':'attention',
				'intermediate':'intermediate',
				'output':'output'
			}
		elif shared_type == 'attention':
			tmp = {
				"layer":"layer_shared",
				'attention':'attention',
				'intermediate':'intermediate_{}'.format(idx),
				'output':'output_{}'.format(idx)
			}
		elif shared_type == 'ffn':
			tmp = {
				"layer":"layer_shared",
				'attention':'attention_{}'.format(idx),
				'intermediate':'intermediate',
				'output':'output'
			}
		else:
			tmp = {
				"layer":"layer_{}".format(idx),
				'attention':'attention',
				'intermediate':'intermediate',
				'output':'output'
			}

		return tmp

	all_layer_outputs = []

	for layer_idx in range(num_hidden_layers):

		idx_scope = layer_scope(layer_idx, shared_type)

		with tf.variable_scope(idx_scope['layer'], reuse=tf.AUTO_REUSE):
			layer_input = prev_output

			with tf.variable_scope(idx_scope['attention'], reuse=tf.AUTO_REUSE):
				attention_heads = []

				with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
					layer_input_pre = layer_norm(layer_input)

				with tf.variable_scope("self"):
					attention_head = attention_layer(
							from_tensor=layer_input_pre,
							to_tensor=layer_input_pre,
							attention_mask=attention_mask,
							num_attention_heads=num_attention_heads,
							size_per_head=attention_head_size,
							attention_probs_dropout_prob=attention_probs_dropout_prob,
							initializer_range=initializer_range,
							do_return_2d_tensor=True,
							batch_size=batch_size,
							from_seq_length=seq_length,
							to_seq_length=seq_length)
					attention_heads.append(attention_head)

				attention_output = None
				if len(attention_heads) == 1:
					attention_output = attention_heads[0]
				else:
					# In the case where we have other sequences, we just concatenate
					# them to the self-attention head before the projection.
					attention_output = tf.concat(attention_heads, axis=-1)

				# Run a linear projection of `hidden_size` then add a residual
				# with `layer_input`.
				with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
					attention_output = tf.layers.dense(
							attention_output,
							hidden_size,
							kernel_initializer=create_initializer(initializer_range))
					attention_output = dropout(attention_output, hidden_dropout_prob)

					# attention_output = layer_norm(attention_output + layer_input)
					attention_output = attention_output + layer_input

			with tf.variable_scope(idx_scope['output'], reuse=tf.AUTO_REUSE):
				attention_output_pre = layer_norm(attention_output)

			# The activation is only applied to the "intermediate" hidden layer.
			with tf.variable_scope(idx_scope['intermediate'], reuse=tf.AUTO_REUSE):
				intermediate_output = tf.layers.dense(
						attention_output_pre,
						intermediate_size,
						activation=intermediate_act_fn,
						kernel_initializer=create_initializer(initializer_range))

			# Down-project back to `hidden_size` then add the residual.
			with tf.variable_scope(idx_scope['output'], reuse=tf.AUTO_REUSE):
				layer_output = tf.layers.dense(
						intermediate_output,
						hidden_size,
						kernel_initializer=create_initializer(initializer_range))
				layer_output = dropout(layer_output, hidden_dropout_prob)

				# layer_output = layer_norm(layer_output + attention_output)
				layer_output = layer_output + attention_output
				prev_output = layer_output
				all_layer_outputs.append(layer_output)

	if do_return_all_layers:
		final_outputs = []
		for layer_output in all_layer_outputs:
			final_output = bert_utils.reshape_from_matrix(layer_output, input_shape)
			final_outputs.append(final_output)
		return final_outputs
	else:
		final_output = bert_utils.reshape_from_matrix(prev_output, input_shape)
		return final_output
```



## 损失函数

### MLM

**主要流程为：**

1. **提取模型输出中mask position位置的向量**
2. **经过变换 每一个输出vocab size大小**
3. **与标签计算交叉熵损失**

```
def get_masked_lm_output(albert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)


  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=albert_config.embedding_size,
          activation=modeling.get_activation(albert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              albert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[albert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=albert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)
```

### SOP

主要流程为：

1. 模型输出向量 转换为 输出为2维的向量
2. 与标签计算交叉熵损失

```
def get_sentence_order_output(albert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, albert_config.hidden_size],
        initializer=modeling.create_initializer(
            albert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, log_probs)
```



# 实验结果

| 模型 | 验证集 | 测试集 |
| :--: | :----: | :----: |
|  roberta    |     0.88503   | 0.86344 |
|  albert    |    0.85662    | 0.84960 |
|   预训练后roberta   | **0.89343** | 0.85328 |
|   预训练后albert   | 0.84958 | **0.85224** |

# 总结

模型根据验证集结果保存最优模型，因此测试集上表现不一定是最优的，我们主要看在验证集上的表现。以上模型在相同参数下只跑了一次，因此结果会略有浮动。

1. roberta在预训练后效果取得提升，经过再次预训练，模型领域与微调领域更加接近，效果更好
2. Albert预训练后效果下降，可能与我们构建数据的方式有关，构建的数据与SOP任务并不符合，可以尝试更符合要求的数据进行测试。

# TO DO

- MACBert-20220319暂未开源预训练代码
- 根据SpanBert改为n-gram 掩码与SBO任务
- Pytorch与keras的预训练
