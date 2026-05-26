# coding=utf-8
"""
标题相似度去重模块

对来自不同平台的相似标题进行去重，保留排名最高的那条。
解决"同一条新闻被多个平台推送"的重复问题。

自动适配两种数据格式：
- 关键词匹配格式：stats 中 titles 为 {source_id: [title_dict, ...]}
- AI 筛选格式：stats 中 titles 为 [title_dict, ...]（平铺列表）
"""

from difflib import SequenceMatcher
from typing import Dict, List, Set, Tuple


def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _get_rank(title_data: Dict) -> int:
    ranks = title_data.get("ranks", [])
    if ranks:
        return min(ranks)
    rank = title_data.get("rank")
    if rank is not None:
        return rank
    rank_threshold = title_data.get("rank_threshold", 999)
    return rank_threshold


def _dedup_sorted_entries(
    entries: List[Dict],
    threshold: float,
    min_title_length: int,
) -> Set[int]:
    to_remove: Set[int] = set()

    for i in range(len(entries)):
        if i in to_remove:
            continue
        title_i = entries[i].get("title", "")
        if not title_i or len(title_i) < min_title_length:
            continue

        for j in range(i + 1, len(entries)):
            if j in to_remove:
                continue
            title_j = entries[j].get("title", "")
            if not title_j or len(title_j) < min_title_length:
                continue

            sim = _title_similarity(title_i, title_j)
            if sim >= threshold:
                to_remove.add(j)

    return to_remove


def dedup_similar_titles(
    stats: List[Dict],
    threshold: float = 0.75,
    min_title_length: int = 4,
) -> List[Dict]:
    """对 stats 中的标题进行跨平台标题相似度去重（自动适配两种格式）。

    去重策略：
    - 同一个 keyword 组内的标题两两比较相似度
    - 相似度 >= threshold 的视为重复
    - 保留排名更靠前的那条

    Args:
        stats: 统计结果列表
        threshold: 相似度阈值，默认 0.75
        min_title_length: 最短标题长度

    Returns:
        去重后的 stats（原地修改并返回）
    """
    for stat in stats:
        titles_data = stat.get("titles", {})
        if not titles_data:
            continue

        if isinstance(titles_data, dict):
            _dedup_dict_format(stat, titles_data, threshold, min_title_length)
        elif isinstance(titles_data, list):
            _dedup_flat_format(stat, titles_data, threshold, min_title_length)

    return stats


def _dedup_dict_format(
    stat: Dict,
    titles_dict: Dict,
    threshold: float,
    min_title_length: int,
) -> None:
    """对 {source_id: [title_dict, ...]} 格式去重"""
    all_entries = []
    for source_id, source_titles in titles_dict.items():
        for title_data in source_titles:
            all_entries.append(title_data)

    if len(all_entries) <= 1:
        return

    all_entries.sort(key=lambda e: (_get_rank(e), len(e.get("title", ""))))

    removed_indices = _dedup_sorted_entries(all_entries, threshold, min_title_length)

    if not removed_indices:
        return

    removed_count = 0
    for source_id in list(titles_dict.keys()):
        kept = []
        for title_data in titles_dict[source_id]:
            try:
                idx = all_entries.index(title_data)
            except ValueError:
                kept.append(title_data)
                continue
            if idx not in removed_indices:
                kept.append(title_data)
            else:
                removed_count += 1
        if kept:
            titles_dict[source_id] = kept
        else:
            del titles_dict[source_id]

    if removed_count > 0:
        stat["count"] = max(0, stat.get("count", 0) - removed_count)
        keyword = stat.get("word", "?")
        print(f"[去重] 「{keyword}」: 移除 {removed_count} 条重复标题")


def _dedup_flat_format(
    stat: Dict,
    titles: List,
    threshold: float,
    min_title_length: int,
) -> None:
    """对 [title_dict, ...] 格式去重"""
    if len(titles) <= 1:
        return

    titles.sort(key=lambda e: (_get_rank(e), len(e.get("title", ""))))

    removed_indices = _dedup_sorted_entries(titles, threshold, min_title_length)

    if not removed_indices:
        return

    kept = [t for i, t in enumerate(titles) if i not in removed_indices]
    stat["titles"] = kept
    removed_count = len(removed_indices)
    stat["count"] = max(0, stat.get("count", 0) - removed_count)
    keyword = stat.get("word", "?")
    print(f"[去重] 「{keyword}」: 移除 {removed_count} 条重复标题")


def dedup_similar_titles_flat(
    stats: List[Dict],
    threshold: float = 0.75,
    min_title_length: int = 4,
) -> List[Dict]:
    """对 AI 筛选格式的 stats 进行标题相似度去重（兼容旧名称，内部委托给 dedup_similar_titles）"""
    return dedup_similar_titles(stats, threshold=threshold, min_title_length=min_title_length)