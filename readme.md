
# 本地运行

- run_mlm_no_trainer.py
- run_mlm_no_trainer_debug.py
- run_mlm_no_trainer_debug_wwm.py

根据官网文档，指定文件路径加载dataset。

wwm指定切词函数，生成全词id文件，利用DataCollatorForWholeWordMask函数完成全词掩码。

# 线上运行

- run_mlm_no_trainer_online.py
- run_mlm_no_trainer_online_read.py

第一种方式，以相对路径方式读取oss文件，优点是简单快捷，缺点是难以处理大规模数据。

第二种方式，以迭代的方式读取odps数据表，加工成相同的dataset类型，直接处理为bert模型的输入数据。

# 分布式运行

- run_mlm_no_trainer_online_read.py
- run_mlm_no_trainer_online_read_DDP.py
- run_mlm_no_trainer_online_read_DDP2.py
- run_mlm_no_trainer_online_read_DDP_span.py

run_mlm_no_trainer_online_read.py 采用DP方式，实现单机多卡，改动简单，修改模型
```
model = torch.nn.DataParallel(model, device_ids=range(6)) #TODO 在使用net = torch.nn.DataParallel(net)之后，原来的net会被封装为新的net的module属性里。
```

run_mlm_no_trainer_online_read_DDP.py 通过torch.multiprocessing自己进行进程管理
特别注意的是：由于多机多卡情况下，需要针对rank，worldsize进行调整，获得每个机器下的localrank

```
    args = parse_args()
    # world_size = 10
    world_size = torch.cuda.device_count()
    print('{}:{}'.format(world_size, '---' * 100))
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
    
    ………………
    
    rank = int(os.environ['RANK']) * nnodes + local_rank
    world_size = nnodes * int(os.environ['WORLD_SIZE'])
```

run_mlm_no_trainer_online_read_DDP2.py 采用DDP方式，实现多机多卡，启动方式使用easypai
核心点在于初始化环境，配置各个机器所使用的rank，及model GPU
```
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank passed from distributed launcher.",
    )
    ...
if __name__ == "__main__":
    args = parse_args()
    dist.init_process_group(backend='nccl')
    dist.barrier()
    args.world_size = dist.get_world_size()
    args.rank = dist.get_rank()
    print("world size:", args.world_size, " rank:", args.rank, " local rank:", args.local_rank)
    main(args)
    
    ......
    model = torch.nn.parallel.DistributedDataParallel(model)
```



run_mlm_no_trainer_online_read_DDP_span.py 采用Span掩码的方式替换原掩码方式，分布式运行仍采用DDP
```
    def _span_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        sym_indexes = set()
        for (i, token) in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]" or token == "[PAD]":
                sym_indexes.add(i)
                continue

        num_to_predict = min(max_predictions, max(1, int(round((len(input_tokens) - len(sym_indexes)) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        sent_length = len(input_tokens)

        while len(covered_indexes) < num_to_predict:
            span_len = np.random.choice(self.lens, p=self.len_distrib) # 随机选择span长度

            anchor = np.random.choice(sent_length)
            if anchor in covered_indexes or anchor in sym_indexes: # 随机生成起点
                continue
            left1 = anchor
            masked_lms.append([left1, left1])
            right1 = min(anchor + span_len, sent_length)
            for i in range(left1, right1):
                if len(covered_indexes) >= num_to_predict:
                    break
                if i in sym_indexes:
                    break
                covered_indexes.add(i)
                masked_lms[-1][-1] = i

        # assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels
```
