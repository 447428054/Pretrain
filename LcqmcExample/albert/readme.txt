若数据处理与训练过程中序列长度不同--max_predictions_per_seq=51，会导致bug：

example_parsing_ops.cc:240 : Invalid argument: Key: masked_lm_weights. Can't parse serialized Example.

1. 每个文件中，一个sentence占一行，不同document之间加一个空行分割
[['有', '人', '知', '道', '叫', '什', '么', '名', '字', '吗', '[UNK]', '？'], ['有', '人', '知', '道', '名', '字', '吗']]

2. 从一个文档中获取sentence,sentece进行全词分词，当长度达到最大长度或者遍历完整个文档了，A[SEP]B 随机分割句子，50%概率交换顺序，得到SOP标签
tokenA:['有', '##人', '知', '##道', '叫', '什', '##么', '名', '##字', '吗', '[UNK]', '？']
tokenB:['有', '##人', '知', '##道', '名', '##字', '吗']

只有一句话的就continue

3. 对获得的token序列，进行掩码:返回 掩码结果，掩码的位置，掩码的标签
tokens:['[CLS]', '有', '人', '知', '道', '叫', '什', '么', '名', '[MASK]', '吗', '[UNK]', '？', '[SEP]', '[MASK]', '人', '知', '[MASK]', '名', '字', '吗', '[SEP]']
masked_lm_positions:[9, 14, 17]
masked_lm_labels:['##字', '有', '##道']
is_random_next:False
