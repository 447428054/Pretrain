# 代码声明

该代码参考苏神bert4keras预训练代码

# 预训练相关代码

目前支持RoBERTa和GPT模式的预训练。请在tensorflow 1.14或1.15下运行。

## 使用
```
python data_utils.py # 生成tfrecord
python pretraining.py # 启动预训练过程
```

请阅读`data_utils.py`和`pretraining.py`修改相应的配置和参数，以适配自己的语料和设备。

## 背景

keras是一个友好的框架，通常我们都是基于tf后端使用，另外还有tf.keras可以使用，基本上跟keras 2.3.x的接口一致了。

这种一致性意味着使用keras几乎就相当于使用tf，这意味着tf的一切优势keras也有，但tf没有的优势（比如使用简便）keras也有。

因此，作者参考原训练过程地实现了基于keras的预训练脚本，而有了这个keras版之后，因为前面所述的一致性，所以我们可以很轻松地迁移到多GPU上训练，也可以很轻松地迁移到TPU上训练。

## 代码改动

代码改动主要涉及数据处理方面，数据处理的修改分为数据读取与掩码生成。

### 数据读取

预训练文本格式如下：`textA\n\ntextB`
```
手机三脚架网红直播支架桌面自拍杆蓝牙遥控三脚架摄影拍摄拍照抖音看电视神器三角架便携伸缩懒人户外支撑架 【女神粉】自带三脚架+蓝牙遥控

牛皮纸袋手提袋定制logo烘焙购物服装包装外卖打包袋子礼品袋纸质 黑色 32*11*25 大横100个

彩色金属镂空鱼尾夹长尾夹 手帐设计绘图文具收纳 夹子 鱼尾夹炫彩大号
```

对应修改**some_texts**函数，完成数据处理。

### 掩码生成

参考SpanBERT，完成ngram掩码

核心代码如下：
```
def __init__(
    self, tokenizer, word_segment, lower=1, upper=10, p=0.3, mask_rate=0.15, sequence_length=512
):
    """参数说明：
        tokenizer必须是bert4keras自带的tokenizer类；
        word_segment是任意分词函数。
    """
    super(TrainingDatasetRoBERTa, self).__init__(tokenizer, sequence_length)
    self.word_segment = word_segment
    self.mask_rate = mask_rate

    self.lower = lower
    self.upper = upper
    self.p = p

    self.lens = list(range(self.lower, self.upper + 1))
    self.len_distrib = [self.p * (1-self.p)**(i - self.lower) for i in range(self.lower, self.upper + 1)] if self.p >= 0 else None
    self.len_distrib = [x / (sum(self.len_distrib)) for x in self.len_distrib]
    print(self.len_distrib, self.lens)

def sentence_process(self, text):
    """单个文本的处理函数
    流程：分词，然后转id，按照mask_rate构建全词mask的序列
          来指定哪些token是否要被mask
    """

    word_tokens = self.tokenizer.tokenize(text=text)[1:-1]
    word_token_ids = self.tokenizer.tokens_to_ids(word_tokens)

    sent_length = len(word_tokens)
    mask_num = math.ceil(sent_length * self.mask_rate)
    mask = set()
    spans = []

    while len(mask) < mask_num:
        span_len = np.random.choice(self.lens, p=self.len_distrib) # 随机选择span长度

        anchor = np.random.choice(sent_length)
        if anchor in mask: # 随机生成起点
            continue
        left1 = anchor
        spans.append([left1, left1])
        right1 = min(anchor + span_len, sent_length)
        for i in range(left1, right1):
            if len(mask) >= mask_num:
                break
            mask.add(i)
            spans[-1][-1] = i

    spans = merge_intervals(spans)
    word_mask_ids = [0] * len(word_tokens)
    for (st, ed) in spans:
        for idx in range(st, ed + 1):
            wid = word_token_ids[idx]
            word_mask_ids[idx] = self.token_process(wid) + 1

    return [word_token_ids, word_mask_ids]
```

### 训练修改

pretraining-lz.py：修改模型保存方式
```
# model.save_weights 保存的模型，用 model.load_weights 加载
# bert.save_weights_as_checkpoint 保存的模型，用 bert.load_weights_from_checkpoint 加载
# 不要问为什么保存的模型用 build_transformer_model 加载不了
# 先搞清楚对应情况，build_transformer_model 是用 load_weights_from_checkpoint 加载的。
```
pretraining-lz-fgm.py：添加对抗训练

trans-model.py：加载模型，之后另存模型，核心代码如下：
```
train_model.load_weights(model_saved_path)
bert.save_weights_as_checkpoint(filename='../save_pretrain/transpretrain-traintestAB-unlabel-nezha-rawbase-ngram-epo26-FGM-epo1/bert_model.ckpt')
```
