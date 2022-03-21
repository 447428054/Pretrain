#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： JMXGODLZZ
# datetime： 2022/3/18 下午2:28 
# ide： PyCharm
import jieba
import re

def get_new_segment(segment): #  新增的方法 ####
    """
    输入一句话，返回一句经过处理的话: 为了支持中文全称mask，将被分开的词，将上特殊标记("#")，使得后续处理模块，能够知道哪些字是属于同一个词的。
    :param segment: 一句话
    :return: 一句处理过的话
    """
    seq_cws = jieba.lcut("".join(segment))
    seq_cws_dict = {x: 1 for x in seq_cws}
    new_segment = []
    i = 0
    while i < len(segment):
        if len(re.findall('[\u4E00-\u9FA5]', segment[i]))==0: # 不是中文的，原文加进去。
            new_segment.append(segment[i])
            i += 1
            continue

        has_add = False
        for length in range(3,0,-1):
            if i+length>len(segment):
                continue
            if ''.join(segment[i:i+length]) in seq_cws_dict:
                new_segment.append(segment[i])
                for l in range(1, length):
                    new_segment.append('##' + segment[i+l])
                i += length
                has_add = True
                break
        if not has_add:
            new_segment.append(segment[i])
            i += 1
    return new_segment

print(get_new_segment('无人机在探查武汉市长江大桥的路况'))