"""
评估指标实现。
"""

import logging
from typing import Dict, Callable, List, Optional, Any
import numpy as np

logger = logging.getLogger("brainforge.evaluation.metrics")


def get_evaluators(metrics: List[str]) -> Dict[str, Callable]:
    """
    获取评估指标函数

    参数:
        metrics: 指标名称列表

    返回:
        评估函数字典
    """
    available_metrics = {
        "accuracy": accuracy_score,
        "relevance": relevance_score,
        "fluency": fluency_score,
        "consistency": consistency_score,
        "coherence": coherence_score
    }

    # 过滤不支持的指标
    evaluators = {}
    for metric in metrics:
        if metric in available_metrics:
            evaluators[metric] = available_metrics[metric]
        else:
            logger.warning(f"Unsupported metric: {metric}")

    return evaluators


def accuracy_score(generated: str, reference: str) -> float:
    """
    计算生成文本与参考文本的准确性得分

    参数:
        generated: 生成的文本
        reference: 参考文本

    返回:
        准确性得分(0-1)
    """
    # 简化实现：基于单词重叠率
    # 实际应用中可使用更复杂的算法
    gen_words = set(generated.lower().split())
    ref_words = set(reference.lower().split())

    if not ref_words:
        return 0.0

    overlap = len(gen_words.intersection(ref_words))
    return min(overlap / len(ref_words), 1.0)


def relevance_score(generated: str, reference: str) -> float:
    """
    计算生成文本的相关性得分

    参数:
        generated: 生成的文本
        reference: 参考文本

    返回:
        相关性得分(0-1)
    """

    # 简化实现：基于单词共现
    # 实际应用中可使用语义相似度算法

    # 提取关键词（简化为频率最高的词）
    def get_keywords(text, top_n=10):
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # 忽略短词
                word_freq[word] = word_freq.get(word, 0) + 1

        # 按频率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return set([w for w, _ in sorted_words[:top_n]])

    gen_keywords = get_keywords(generated)
    ref_keywords = get_keywords(reference)

    if not ref_keywords:
        return 0.0

    overlap = len(gen_keywords.intersection(ref_keywords))
    return min(overlap / len(ref_keywords), 1.0)


def fluency_score(generated: str, reference: str = None) -> float:
    """
    计算生成文本的流畅度得分

    参数:
        generated: 生成的文本
        reference: 参考文本(不使用)

    返回:
        流畅度得分(0-1)
    """
    # 简化实现：基于句子长度和标点符号
    # 实际应用中可使用语言模型困惑度

    if not generated:
        return 0.0

    # 分割成句子
    sentences = generated.split(".")
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    # 计算平均句子长度
    avg_len = sum(len(s.split()) for s in sentences) / len(sentences)

    # 句子长度得分(假设理想长度为10-20词)
    len_score = 1.0 - min(abs(avg_len - 15) / 15, 1.0)

    # 标点符号比例(假设理想比例为5-10%)
    punct_count = sum(1 for c in generated if c in ",.;:!?")
    punct_ratio = punct_count / len(generated)
    punct_score = 1.0 - min(abs(punct_ratio - 0.075) / 0.075, 1.0)

    # 综合得分
    return 0.7 * len_score + 0.3 * punct_score


def consistency_score(generated: str, reference: str) -> float:
    """
    计算生成文本的一致性得分

    参数:
        generated: 生成的文本
        reference: 参考文本

    返回:
        一致性得分(0-1)
    """

    # 简化实现：基于关键事实的提取和比较
    # 实际应用中可使用更复杂的事实提取和验证算法

    # 提取数字和专有名词(简化为大写单词)
    def extract_facts(text):
        words = text.split()

        # 提取数字
        numbers = [w for w in words if any(c.isdigit() for c in w)]

        # 提取专有名词(简化为大写开头的词)
        proper_nouns = [w for w in words if w and w[0].isupper() and w not in ["I", "A", "The"]]

        return set(numbers + proper_nouns)

    gen_facts = extract_facts(generated)
    ref_facts = extract_facts(reference)

    if not ref_facts:
        return 1.0  # 没有需要一致的事实

    # 计算事实匹配率
    matches = 0
    for fact in ref_facts:
        if fact in gen_facts or any(fact in w for w in gen_facts):
            matches += 1

    return matches / len(ref_facts)


def coherence_score(generated: str, reference: str = None) -> float:
    """
    计算生成文本的连贯性得分

    参数:
        generated: 生成的文本
        reference: 参考文本(不使用)

    返回:
        连贯性得分(0-1)
    """
    # 简化实现：基于连接词和代词的使用
    # 实际应用中可使用更复杂的连贯性分析

    if not generated:
        return 0.0

    # 分割成句子
    sentences = generated.split(".")
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return 0.5  # 单句无法完全评估连贯性

    # 连接词列表
    connectives = [
        "also", "however", "therefore", "thus", "furthermore",
        "moreover", "nevertheless", "meanwhile", "consequently",
        "in addition", "on the other hand", "as a result"
    ]

    # 代词列表
    pronouns = [
        "it", "they", "them", "their", "these", "those",
        "this", "that", "which", "who", "whom"
    ]

    # 计算连接词和代词的使用
    conn_count = sum(1 for s in sentences if any(c in s.lower() for c in connectives))
    pron_count = sum(1 for s in sentences if any(p in s.lower().split() for p in pronouns))

    # 理想的连接词比例(假设30-60%的句子应包含连接词)
    conn_score = 1.0 - min(abs(conn_count / len(sentences) - 0.45) / 0.45, 1.0)

    # 理想的代词比例(假设40-70%的句子应包含代词)
    pron_score = 1.0 - min(abs(pron_count / len(sentences) - 0.55) / 0.55, 1.0)

    # 综合得分
    return 0.6 * conn_score + 0.4 * pron_score