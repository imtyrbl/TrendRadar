# coding=utf-8
"""
应用上下文模块

提供配置上下文类，封装所有依赖配置的操作，消除全局状态和包装函数。
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from trendradar.utils.time import (
    DEFAULT_TIMEZONE,
    get_configured_time,
    format_date_folder,
    format_time_filename,
    get_current_time_display,
    convert_time_for_display,
    format_iso_time_friendly,
    is_within_days,
)
from trendradar.core import (
    load_frequency_words,
    matches_word_groups,
    read_all_today_titles,
    detect_latest_new_titles,
    count_word_frequency,
    Scheduler,
)
from trendradar.core.loader import resolve_ai_config
from trendradar.report import (
    prepare_report_data,
    generate_html_report,
    render_html_content,
)
from trendradar.notification import (
    render_feishu_content,
    render_dingtalk_content,
    split_content_into_batches,
    NotificationDispatcher,
)
from trendradar.ai import AITranslator
from trendradar.ai.filter import AIFilterResult
from trendradar.ai.filter_pipeline import AIFilterPipeline, _TagExtractionError
from trendradar.core.dedup import dedup_similar_titles_flat
from trendradar.storage import get_storage_manager


class AppContext:
    """
    应用上下文类

    封装所有依赖配置的操作，提供统一的接口。
    消除对全局 CONFIG 的依赖，提高可测试性。

    使用示例:
        config = load_config()
        ctx = AppContext(config)

        # 时间操作
        now = ctx.get_time()
        date_folder = ctx.format_date()

        # 存储操作
        storage = ctx.get_storage_manager()

        # 报告生成
        html = ctx.generate_html_report(stats, total_titles, ...)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化应用上下文

        Args:
            config: 完整的配置字典
        """
        self.config = config
        self._storage_manager = None
        self._scheduler = None

    # === 配置访问 ===

    @property
    def timezone(self) -> str:
        """获取配置的时区"""
        return self.config.get("TIMEZONE", DEFAULT_TIMEZONE)

    @property
    def rank_threshold(self) -> int:
        """获取排名阈值"""
        return self.config.get("RANK_THRESHOLD", 50)

    @property
    def weight_config(self) -> Dict:
        """获取权重配置"""
        return self.config.get("WEIGHT_CONFIG", {})

    @property
    def platforms(self) -> List[Dict]:
        """获取平台配置列表"""
        return self.config.get("PLATFORMS", [])

    @property
    def platform_ids(self) -> List[str]:
        """获取平台ID列表"""
        return [p["id"] for p in self.platforms]

    @property
    def rss_config(self) -> Dict:
        """获取 RSS 配置"""
        return self.config.get("RSS", {})

    @property
    def rss_enabled(self) -> bool:
        """RSS 是否启用"""
        return self.rss_config.get("ENABLED", False)

    @property
    def rss_feeds(self) -> List[Dict]:
        """获取 RSS 源列表"""
        return self.rss_config.get("FEEDS", [])

    @property
    def display_mode(self) -> str:
        """获取显示模式 (keyword | platform)"""
        return self.config.get("DISPLAY_MODE", "keyword")

    @property
    def show_new_section(self) -> bool:
        """是否显示新增热点区域"""
        return self.config.get("DISPLAY", {}).get("REGIONS", {}).get("NEW_ITEMS", True)

    @property
    def region_order(self) -> List[str]:
        """获取区域显示顺序"""
        default_order = ["hotlist", "rss", "new_items", "standalone", "ai_analysis"]
        return self.config.get("DISPLAY", {}).get("REGION_ORDER", default_order)

    @property
    def filter_method(self) -> str:
        """获取筛选策略: keyword | ai"""
        return self.config.get("FILTER", {}).get("METHOD", "keyword")

    @property
    def ai_priority_sort_enabled(self) -> bool:
        """AI 模式标签排序开关（与 keyword 的 sort_by_position_first 解耦）"""
        return self.config.get("FILTER", {}).get("PRIORITY_SORT_ENABLED", False)

    @property
    def ai_filter_config(self) -> Dict:
        """获取 AI 筛选配置"""
        return self.config.get("AI_FILTER", {})

    @property
    def ai_filter_enabled(self) -> bool:
        """AI 筛选是否启用（基于 filter.method 判断）"""
        return self.filter_method == "ai"

    # === 时间操作 ===

    def get_time(self) -> datetime:
        """获取当前配置时区的时间"""
        return get_configured_time(self.timezone)

    def format_date(self) -> str:
        """格式化日期文件夹 (YYYY-MM-DD)"""
        return format_date_folder(timezone=self.timezone)

    def format_time(self) -> str:
        """格式化时间文件名 (HH-MM)"""
        return format_time_filename(self.timezone)

    def get_time_display(self) -> str:
        """获取时间显示 (HH:MM)"""
        return get_current_time_display(self.timezone)

    @staticmethod
    def convert_time_display(time_str: str) -> str:
        """将 HH-MM 转换为 HH:MM"""
        return convert_time_for_display(time_str)

    # === 存储操作 ===

    def get_storage_manager(self):
        """获取存储管理器（延迟初始化，单例）"""
        if self._storage_manager is None:
            storage_config = self.config.get("STORAGE", {})
            remote_config = storage_config.get("REMOTE", {})
            local_config = storage_config.get("LOCAL", {})
            pull_config = storage_config.get("PULL", {})

            self._storage_manager = get_storage_manager(
                backend_type=storage_config.get("BACKEND", "auto"),
                data_dir=local_config.get("DATA_DIR", "output"),
                enable_txt=storage_config.get("FORMATS", {}).get("TXT", True),
                enable_html=storage_config.get("FORMATS", {}).get("HTML", True),
                remote_config={
                    "bucket_name": remote_config.get("BUCKET_NAME", ""),
                    "access_key_id": remote_config.get("ACCESS_KEY_ID", ""),
                    "secret_access_key": remote_config.get("SECRET_ACCESS_KEY", ""),
                    "endpoint_url": remote_config.get("ENDPOINT_URL", ""),
                    "region": remote_config.get("REGION", ""),
                },
                local_retention_days=local_config.get("RETENTION_DAYS", 0),
                remote_retention_days=remote_config.get("RETENTION_DAYS", 0),
                pull_enabled=pull_config.get("ENABLED", False),
                pull_days=pull_config.get("DAYS", 7),
                timezone=self.timezone,
            )
        return self._storage_manager

    def get_output_path(self, subfolder: str, filename: str) -> str:
        """获取输出路径（扁平化结构：output/类型/日期/文件名）"""
        output_dir = Path("output") / subfolder / self.format_date()
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir / filename)

    # === 数据处理 ===

    def read_today_titles(
        self, platform_ids: Optional[List[str]] = None, quiet: bool = False
    ) -> Tuple[Dict, Dict, Dict]:
        """读取当天所有标题"""
        return read_all_today_titles(self.get_storage_manager(), platform_ids, quiet=quiet)

    def detect_new_titles(
        self, platform_ids: Optional[List[str]] = None, quiet: bool = False
    ) -> Dict:
        """检测最新批次的新增标题"""
        return detect_latest_new_titles(self.get_storage_manager(), platform_ids, quiet=quiet)

    def is_first_crawl(self) -> bool:
        """检测是否是当天第一次爬取"""
        return self.get_storage_manager().is_first_crawl_today()

    # === 频率词处理 ===

    def load_frequency_words(
        self, frequency_file: Optional[str] = None
    ) -> Tuple[List[Dict], List[str], List[str]]:
        """加载频率词配置"""
        return load_frequency_words(frequency_file)

    def matches_word_groups(
        self,
        title: str,
        word_groups: List[Dict],
        filter_words: List[str],
        global_filters: Optional[List[str]] = None,
    ) -> bool:
        """检查标题是否匹配词组规则"""
        return matches_word_groups(title, word_groups, filter_words, global_filters)

    # === 统计分析 ===

    def count_frequency(
        self,
        results: Dict,
        word_groups: List[Dict],
        filter_words: List[str],
        id_to_name: Dict,
        title_info: Optional[Dict] = None,
        new_titles: Optional[Dict] = None,
        mode: str = "daily",
        global_filters: Optional[List[str]] = None,
        quiet: bool = False,
    ) -> Tuple[List[Dict], int]:
        """统计词频"""
        return count_word_frequency(
            results=results,
            word_groups=word_groups,
            filter_words=filter_words,
            id_to_name=id_to_name,
            title_info=title_info,
            rank_threshold=self.rank_threshold,
            new_titles=new_titles,
            mode=mode,
            global_filters=global_filters,
            weight_config=self.weight_config,
            max_news_per_keyword=self.config.get("MAX_NEWS_PER_KEYWORD", 0),
            sort_by_position_first=self.config.get("SORT_BY_POSITION_FIRST", False),
            is_first_crawl_func=self.is_first_crawl,
            convert_time_func=self.convert_time_display,
            quiet=quiet,
        )

    # === 报告生成 ===

    def prepare_report(
        self,
        stats: List[Dict],
        failed_ids: Optional[List] = None,
        new_titles: Optional[Dict] = None,
        id_to_name: Optional[Dict] = None,
        mode: str = "daily",
        frequency_file: Optional[str] = None,
    ) -> Dict:
        """准备报告数据"""
        return prepare_report_data(
            stats=stats,
            failed_ids=failed_ids,
            new_titles=new_titles,
            id_to_name=id_to_name,
            mode=mode,
            rank_threshold=self.rank_threshold,
            show_new_section=self.show_new_section,
        )

    def generate_html(
        self,
        stats: List[Dict],
        total_titles: int,
        failed_ids: Optional[List] = None,
        new_titles: Optional[Dict] = None,
        id_to_name: Optional[Dict] = None,
        mode: str = "daily",
        update_info: Optional[Dict] = None,
        rss_items: Optional[List[Dict]] = None,
        rss_new_items: Optional[List[Dict]] = None,
        ai_analysis: Optional[Any] = None,
        standalone_data: Optional[Dict] = None,
        frequency_file: Optional[str] = None,
        report_metadata: Optional[Dict] = None,
        translate_report_func: Optional[Any] = None,
    ) -> str:
        """生成HTML报告"""
        return generate_html_report(
            stats=stats,
            total_titles=total_titles,
            failed_ids=failed_ids,
            new_titles=new_titles,
            id_to_name=id_to_name,
            mode=mode,
            update_info=update_info,
            rank_threshold=self.rank_threshold,
            output_dir="output",
            date_folder=self.format_date(),
            time_filename=self.format_time(),
            render_html_func=lambda *args, **kwargs: self.render_html(*args, rss_items=rss_items, rss_new_items=rss_new_items, ai_analysis=ai_analysis, standalone_data=standalone_data, **kwargs),
            report_metadata=report_metadata,
            translate_report_func=translate_report_func,
        )

    def render_html(
        self,
        report_data: Dict,
        total_titles: int,
        mode: str = "daily",
        update_info: Optional[Dict] = None,
        rss_items: Optional[List[Dict]] = None,
        rss_new_items: Optional[List[Dict]] = None,
        ai_analysis: Optional[Any] = None,
        standalone_data: Optional[Dict] = None,
    ) -> str:
        """渲染HTML内容"""
        return render_html_content(
            report_data=report_data,
            total_titles=total_titles,
            mode=mode,
            update_info=update_info,
            region_order=self.region_order,
            get_time_func=self.get_time,
            rss_items=rss_items,
            rss_new_items=rss_new_items,
            display_mode=self.display_mode,
            ai_analysis=ai_analysis,
            show_new_section=self.show_new_section,
            standalone_data=standalone_data,
        )

    # === 通知内容渲染 ===

    def render_feishu(
        self,
        report_data: Dict,
        update_info: Optional[Dict] = None,
        mode: str = "daily",
    ) -> str:
        """渲染飞书内容"""
        return render_feishu_content(
            report_data=report_data,
            update_info=update_info,
            mode=mode,
            separator=self.config.get("FEISHU_MESSAGE_SEPARATOR", "---"),
            region_order=self.region_order,
            get_time_func=self.get_time,
            show_new_section=self.show_new_section,
        )

    def render_dingtalk(
        self,
        report_data: Dict,
        update_info: Optional[Dict] = None,
        mode: str = "daily",
    ) -> str:
        """渲染钉钉内容"""
        return render_dingtalk_content(
            report_data=report_data,
            update_info=update_info,
            mode=mode,
            region_order=self.region_order,
            get_time_func=self.get_time,
            show_new_section=self.show_new_section,
        )

    def split_content(
        self,
        report_data: Dict,
        format_type: str,
        update_info: Optional[Dict] = None,
        max_bytes: Optional[int] = None,
        mode: str = "daily",
        rss_items: Optional[list] = None,
        rss_new_items: Optional[list] = None,
        ai_content: Optional[str] = None,
        standalone_data: Optional[Dict] = None,
        ai_stats: Optional[Dict] = None,
        report_type: str = "热点分析报告",
    ) -> List[str]:
        """分批处理消息内容（支持热榜+RSS合并+AI分析+独立展示区）

        Args:
            report_data: 报告数据
            format_type: 格式类型
            update_info: 更新信息
            max_bytes: 最大字节数
            mode: 报告模式
            rss_items: RSS 统计条目列表
            rss_new_items: RSS 新增条目列表
            ai_content: AI 分析内容（已渲染的字符串）
            standalone_data: 独立展示区数据
            ai_stats: AI 分析统计数据
            report_type: 报告类型

        Returns:
            分批后的消息内容列表
        """
        return split_content_into_batches(
            report_data=report_data,
            format_type=format_type,
            update_info=update_info,
            max_bytes=max_bytes,
            mode=mode,
            batch_sizes={
                "dingtalk": self.config.get("DINGTALK_BATCH_SIZE", 20000),
                "feishu": self.config.get("FEISHU_BATCH_SIZE", 29000),
                "default": self.config.get("MESSAGE_BATCH_SIZE", 4000),
            },
            feishu_separator=self.config.get("FEISHU_MESSAGE_SEPARATOR", "---"),
            region_order=self.region_order,
            get_time_func=self.get_time,
            rss_items=rss_items,
            rss_new_items=rss_new_items,
            timezone=self.config.get("TIMEZONE", DEFAULT_TIMEZONE),
            display_mode=self.display_mode,
            ai_content=ai_content,
            standalone_data=standalone_data,
            rank_threshold=self.rank_threshold,
            ai_stats=ai_stats,
            report_type=report_type,
            show_new_section=self.show_new_section,
        )

    # === 通知发送 ===

    def create_notification_dispatcher(self) -> NotificationDispatcher:
        """创建通知调度器"""
        # 创建翻译器（如果启用）
        translator = None
        trans_config = self.config.get("AI_TRANSLATION", {})
        if trans_config.get("ENABLED", False):
            ai_config = resolve_ai_config(self.config, "AI_TRANSLATION_MODEL", "AI_TRANSLATION")
            translator = AITranslator(trans_config, ai_config)

        return NotificationDispatcher(
            config=self.config,
            get_time_func=self.get_time,
            split_content_func=self.split_content,
            translator=translator,
        )

    def create_scheduler(self) -> Scheduler:
        """
        创建调度器（延迟初始化，单例）

        基于 config.yaml 的 schedule 段 + timeline.yaml 构建。
        """
        if self._scheduler is None:
            schedule_config = self.config.get("SCHEDULE", {})
            timeline_data = self.config.get("_TIMELINE_DATA", {})

            self._scheduler = Scheduler(
            schedule_config=schedule_config,
            timeline_data=timeline_data,
            storage_backend=self.get_storage_manager(),
            get_time_func=self.get_time,
            fallback_report_mode=self.config.get("REPORT_MODE", "current"),
            fallback_rss_report_mode=self.config.get("RSS_REPORT_MODE", "current"),
        )
        return self._scheduler

    # === AI 智能筛选 ===

    def _get_ai_filter_pipeline(self) -> "AIFilterPipeline":
        return AIFilterPipeline(
            config=self.config,
            storage_manager=self.get_storage_manager(),
            get_time_func=self.get_time,
        )

    def run_ai_filter(self, interests_file: Optional[str] = None) -> Optional[AIFilterResult]:
        """执行 AI 智能筛选完整流程"""
        if not self.ai_filter_enabled:
            return None
        pipeline = self._get_ai_filter_pipeline()
        # 保留本地自定义：使用 resolve_ai_config 解析 AI_FILTER 专用配置
        pipeline._ai_config = resolve_ai_config(self.config, "AI_FILTER_MODEL", "AI_FILTER")
        try:
            return pipeline.run(interests_file)
        except _TagExtractionError:
            return AIFilterResult(success=False, error="标签提取失败")

    def convert_ai_filter_to_report_data(
        self,
        ai_filter_result: AIFilterResult,
        hotlist_mode: str = "daily",
        rss_mode: str = "daily",
        new_titles: Optional[Dict] = None,
        rss_new_urls: Optional[set] = None,
    ) -> tuple:
        """
        将 AI 筛选结果转换为与关键词匹配相同的数据结构

        AIFilterResult.tags 中每个 tag 对应一个 "word"（关键词组）。
        tag.items 中 source_type="hotlist" 的条目进入热榜 stats，
        source_type="rss" 的条目进入 rss_items stats。

        热榜与 RSS 使用独立的模式参数，以支持 RSS 独立模式。

        Args:
            ai_filter_result: AI 筛选结果
            hotlist_mode: 热榜报告模式 ("daily" | "current" | "incremental")
            rss_mode: RSS 报告模式 ("daily" | "current" | "incremental")
            new_titles: 热榜新增标题 {source_id: {title: data}}，用于 is_new 检测
            rss_new_urls: 新增 RSS 条目的 URL 集合，用于 is_new 检测

        Returns:
            (hotlist_stats, rss_stats, rss_new_stats):
            - hotlist_stats: 与 count_word_frequency() 产出格式一致
            - rss_stats: 与 rss_items 格式一致
            - rss_new_stats: RSS 新增区（rss_stats 中 is_new 的子集），与 rss_new_items 格式一致
        """
        hotlist_stats = []
        rss_stats = []
        rss_new_stats = []  # AI 筛选的 RSS 新增区（is_new 子集），与关键词路径 rss_new_stats 对齐
        max_news = self.config.get("MAX_NEWS_PER_KEYWORD", 0)
        min_score = self.ai_filter_config.get("MIN_SCORE", 0)

        # current 模式：计算最新时间，只保留当前在榜的热榜新闻
        # 与 count_word_frequency(mode="current") 的过滤逻辑对齐
        latest_time = None
        if hotlist_mode == "current":
            for tag_data in ai_filter_result.tags:
                for item in tag_data.get("items", []):
                    if item.get("source_type", "hotlist") == "hotlist":
                        last_time = item.get("last_time", "")
                        if last_time and (latest_time is None or last_time > latest_time):
                            latest_time = last_time
            if latest_time:
                print(f"[AI筛选] current 模式：最新时间 {latest_time}，过滤已下榜新闻")

        # RSS 新鲜度过滤配置（与推送阶段一致）
        rss_config = self.rss_config
        freshness_config = rss_config.get("FRESHNESS_FILTER", {})
        freshness_enabled = freshness_config.get("ENABLED", True)
        default_max_age_days = freshness_config.get("MAX_AGE_DAYS", 3)
        timezone = self.config.get("TIMEZONE", DEFAULT_TIMEZONE)

        feed_max_age_map = {}
        for feed_cfg in self.rss_feeds:
            feed_id = feed_cfg.get("id", "")
            max_age = feed_cfg.get("max_age_days")
            if max_age is not None:
                try:
                    feed_max_age_map[feed_id] = int(max_age)
                except (ValueError, TypeError):
                    pass

        filtered_count = 0
        for tag_data in ai_filter_result.tags:
            tag_name = tag_data.get("tag", "")
            items = tag_data.get("items", [])
            if not items:
                continue

            hotlist_titles = []
            rss_titles = []

            for item in items:
                source_type = item.get("source_type", "hotlist")

                # current 模式：跳过已下榜的热榜新闻
                if hotlist_mode == "current" and latest_time and source_type == "hotlist":
                    if item.get("last_time", "") != latest_time:
                        filtered_count += 1
                        continue

                # 分数阈值过滤：跳过相关度低于 min_score 的新闻
                if min_score > 0:
                    score = item.get("relevance_score", 0)
                    if score < min_score:
                        continue

                # 构建时间显示
                first_time = item.get("first_time", "")
                last_time = item.get("last_time", "")
                if source_type == "rss":
                    # RSS 新鲜度过滤：跳过超过 max_age_days 的旧文章
                    if freshness_enabled and first_time:
                        feed_id = item.get("source_id", "")
                        max_days = feed_max_age_map.get(feed_id, default_max_age_days)
                        if max_days > 0 and not is_within_days(first_time, max_days, timezone):
                            continue

                    # RSS 条目：first_time 是 ISO 格式，用友好格式显示
                    if first_time:
                        time_display = format_iso_time_friendly(first_time, timezone, include_date=True)
                    else:
                        time_display = ""
                else:
                    # 热榜条目：使用 [HH:MM ~ HH:MM] 格式（与 keyword 模式一致）
                    if first_time and last_time and first_time != last_time:
                        first_display = convert_time_for_display(first_time)
                        last_display = convert_time_for_display(last_time)
                        time_display = f"[{first_display} ~ {last_display}]"
                    elif first_time:
                        time_display = convert_time_for_display(first_time)
                    else:
                        time_display = ""

                # 计算 is_new（与 keyword 模式 core/analyzer.py:335-342 对齐）
                if source_type == "rss":
                    is_new = False
                    if rss_new_urls:
                        item_url = item.get("url", "")
                        is_new = item_url in rss_new_urls if item_url else False
                else:
                    is_new = False
                    if new_titles:
                        item_source_id = item.get("source_id", "")
                        item_title = item.get("title", "")
                        if item_source_id in new_titles:
                            is_new = item_title in new_titles[item_source_id]

                # incremental 模式下仅保留本轮新增命中的条目。
                # run_ai_filter() 返回的是 active 结果集合，因此这里需要
                # 显式过滤掉历史已命中的旧条目，才能与 keyword 模式行为对齐。
                # 注意：热榜用 hotlist_mode，RSS 用 rss_mode 独立判断！
                if source_type == "rss":
                    if rss_mode == "incremental" and not is_new:
                        continue
                else:
                    if hotlist_mode == "incremental" and not is_new:
                        continue

                title_entry = {
                    "title": item.get("title", ""),
                    "source_name": item.get("source_name", ""),
                    "url": item.get("url", ""),
                    "mobile_url": item.get("mobile_url", ""),
                    "ranks": item.get("ranks", []),
                    "rank_threshold": self.rank_threshold,
                    "count": item.get("count", 1),
                    "is_new": is_new,
                    "time_display": time_display,
                    "matched_keyword": tag_name,
                }

                if source_type == "rss":
                    rss_titles.append(title_entry)
                else:
                    hotlist_titles.append(title_entry)

            if hotlist_titles:
                if max_news > 0:
                    hotlist_titles = hotlist_titles[:max_news]
                hotlist_stats.append({
                    "word": tag_name,
                    "count": len(hotlist_titles),
                    "position": tag_data.get("position", 9999),
                    "titles": hotlist_titles,
                })

            if rss_titles:
                if max_news > 0:
                    rss_titles = rss_titles[:max_news]
                rss_stats.append({
                    "word": tag_name,
                    "count": len(rss_titles),
                    "position": tag_data.get("position", 9999),
                    "titles": rss_titles,
                })
                # 新增 RSS 区：仅保留本轮新增（is_new）的条目，供推送/HTML 的"RSS 新增"区块使用
                new_rss_titles = [t for t in rss_titles if t.get("is_new")]
                if new_rss_titles:
                    rss_new_stats.append({
                        "word": tag_name,
                        "count": len(new_rss_titles),
                        "position": tag_data.get("position", 9999),
                        "titles": new_rss_titles,
                    })

        if hotlist_mode == "current" and filtered_count > 0:
            total_kept = sum(s["count"] for s in hotlist_stats)
            print(f"[AI筛选] current 模式：过滤 {filtered_count} 条已下榜新闻，保留 {total_kept} 条当前在榜")

        if min_score > 0:
            hotlist_kept = sum(s["count"] for s in hotlist_stats)
            rss_kept = sum(s["count"] for s in rss_stats)
            total_kept = hotlist_kept + rss_kept
            parts = [f"热榜 {hotlist_kept} 条"]
            if rss_kept > 0:
                parts.append(f"RSS {rss_kept} 条")
            print(f"[AI筛选] 分数过滤：min_score={min_score}，保留 {total_kept} 条 score≥{min_score} ({', '.join(parts)})")

        priority_sort_enabled = self.ai_priority_sort_enabled
        if priority_sort_enabled:
            hotlist_stats.sort(key=lambda x: (x.get("position", 9999), -x["count"], x["word"]))
            rss_stats.sort(key=lambda x: (x.get("position", 9999), -x["count"], x["word"]))
            rss_new_stats.sort(key=lambda x: (x.get("position", 9999), -x["count"], x["word"]))
        else:
            hotlist_stats.sort(key=lambda x: (-x["count"], x.get("position", 9999), x["word"]))
            rss_stats.sort(key=lambda x: (-x["count"], x.get("position", 9999), x["word"]))
            rss_new_stats.sort(key=lambda x: (-x["count"], x.get("position", 9999), x["word"]))

        return hotlist_stats, rss_stats, rss_new_stats

    # === 资源清理 ===

    def cleanup(self):
        """清理资源"""
        if self._storage_manager:
            self._storage_manager.cleanup_old_data()
            self._storage_manager.cleanup()
            self._storage_manager = None
