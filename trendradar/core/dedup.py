# coding=utf-8
"""
标题相似度去重模块

对来自不同平台的相似标题进行去重，保留排名最高的那条。
解决"同一条新闻被多个平台推送"的重复问题。

支持两种数据格式：
- 关键词匹配格式：stats 中 titles 为 {source_id: [title_dict, ...]}
- AI 筛选格式：stats 中 titles 为 [title_dict, ...]（平铺列表）
"""

from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple


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


def _dedup_entries(
    entries: List[Tuple[int, Dict]],
    threshold: float,
    min_title_length: int,
) -> List[int]:
    to_remove = []
    skip = set()

    for i in range(len(entries)):
        if i in skip:
            continue
        idx_i, data_i = entries[i]
        title_i = data_i.get("title", "")
        if not title_i or len(title_i) < min_title_length:
            continue

        for j in range(i + 1, len(entries)):
            if j in skip:
                continue
            idx_j, data_j = entries[j]
            title_j = data_j.get("title", "")
            if not title_j or len(title_j) < min_title_length:
                continue

            sim = _title_similarity(title_i, title_j)
            if sim >= threshold:
                to_remove.append(idx_j)
                skip.add(j)

    return to_remove


def dedup_similar_titles(
    stats: List[Dict],
    threshold: float = 0.75,
    min_title_length: int = 4,
) -> List[Dict]:
    """对关键词匹配格式的 stats 进行跨平台标题相似度去重。

    titles 格式：{source_id: [title_dict, ...]}
    """
    for stat in stats:
        titles_dict = stat.get("titles", {})
        if not titles_dict:
            continue

        all_entries = []
        for source_id, source_titles in titles_dict.items():
            for title_data in source_titles:
                all_entries.append(title_data)

        if len(all_entries) <= 1:
            continue

        indexed = list(enumerate(all_entries))
        indexed.sort(key=lambda e: (_get_rank(e[1]), len(e[1].get("title", ""))))

        removed_indices = set(_dedup_entries(indexed, threshold, min_title_length))

        if not removed_indices:
            continue

        removed_count = 0
        for source_id in list(titles_dict.keys()):
            kept = []
            for title_data in titles_dict[source_id]:
                if all_entries.index(title_data) not in removed_indices:
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

    return stats


def dedup_similar_titles_flat(
    stats: List[Dict],
    threshold: float = 0.75,
    min_title_length: int = 4,
) -> List[Dict]:
    """对 AI 筛选格式的 stats 进行标题相似度去重。

    titles 格式：[title_dict, ...]
    """
    for stat in stats:
        titles = stat.get("titles", [])
        if not isinstance(titles, list) or len(titles) <= 1:
            continue

        indexed = list(enumerate(titles))
        indexed.sort(key=lambda e: (_get_rank(e[1]), len(e[1].get("title", ""))))

        removed_indices = set(_dedup_entries(indexed, threshold, min_title_length))

        if not removed_indices:
            continue

        kept = [t for i, t in enumerate(titles) if i not in removed_indices]
        stat["titles"] = kept
        removed_count = len(removed_indices)
        stat["count"] = max(0, stat.get("count", 0) - removed_count)
        keyword = stat.get("word", "?")
        print(f"[去重] 「{keyword}」: 移除 {removed_count} 条重复标题")

    return stats