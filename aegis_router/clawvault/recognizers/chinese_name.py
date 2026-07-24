"""中文人名识别器

基于 spaCy NER 模型 + 百家姓前缀增强的中文人名识别。
当 spaCy 中文模型不可用时，回退到基于百家姓正则的启发式匹配。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from presidio_analyzer import EntityRecognizer, RecognizerResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 百家姓 (Top ~100 常见姓氏)
# ---------------------------------------------------------------------------

_COMMON_SURNAMES: set[str] = {
    "赵", "钱", "孙", "李", "周", "吴", "郑", "王", "冯", "陈",
    "褚", "卫", "蒋", "沈", "韩", "杨", "朱", "秦", "尤", "许",
    "何", "吕", "施", "张", "孔", "曹", "严", "华", "金", "魏",
    "陶", "姜", "戚", "谢", "邹", "喻", "柏", "水", "窦", "章",
    "云", "苏", "潘", "葛", "奚", "范", "彭", "郎", "鲁", "韦",
    "昌", "马", "苗", "凤", "花", "方", "俞", "任", "袁", "柳",
    "酆", "鲍", "史", "唐", "费", "廉", "岑", "薛", "雷", "贺",
    "倪", "汤", "滕", "殷", "罗", "毕", "郝", "邬", "安", "常",
    "乐", "于", "时", "傅", "皮", "卞", "齐", "康", "伍", "余",
    "元", "卜", "顾", "孟", "平", "黄", "穆", "萧", "尹", "姚",
    "邵", "湛", "汪", "祁", "毛", "禹", "狄", "米", "贝", "明",
    "臧", "计", "伏", "成", "戴", "谈", "宋", "茅", "庞", "熊",
    "纪", "舒", "屈", "项", "祝", "董", "梁", "杜", "阮", "蓝",
    "闵", "席", "季", "麻", "强", "贾", "路", "娄", "危", "江",
    "童", "颜", "郭", "梅", "盛", "林", "刁", "钟", "徐", "邱",
    "骆", "高", "夏", "蔡", "田", "樊", "胡", "凌", "霍", "虞",
    "万", "支", "柯", "昝", "管", "卢", "莫", "经", "房", "裘",
    "缪", "干", "解", "应", "宗", "丁", "宣", "邓", "郁", "单",
    "杭", "洪", "包", "诸", "左", "石", "崔", "吉", "钮", "龚",
    "程", "嵇", "邢", "滑", "裴", "陆", "荣", "翁", "荀", "羊",
    # 复姓
    "欧阳", "太史", "端木", "上官", "司马", "东方", "独孤", "南宫",
    "万俟", "闻人", "夏侯", "诸葛", "尉迟", "公羊", "赫连", "澹台",
    "皇甫", "宗政", "濮阳", "公冶", "太叔", "申屠", "公孙", "慕容",
    "仲孙", "钟离", "长孙", "宇文", "司徒", "鲜于", "司空", "令狐",
}

# 上下文关键词 — 出现这些词时增强识别置信度
_CONTEXT_WORDS: list[str] = ["姓名", "名字", "先生", "女士", "name"]

# 中文字符范围
_CJK_CHAR = r"[\u4e00-\u9fff]"

# 常见汉语虚词 / 动词 / 助词 — 这些字不太可能出现在给定名中
# 用于判断名字边界
_NON_NAME_CHARS: set[str] = set(
    "的了是在有不这那我你他她它们也都就和与及"
    "到把被给让对从向往跟比因为所以但可能会要"
    "说做看来去想吃喝打算觉得知道认为告诉问答"
    "又再还已经正在将要于之而且或者如果虽然因此"
    "吗呢吧啊哦嗯呀哈嘛哪谁什么怎么为什么"
    "上下左右前后里外中间旁边东西南北"
    "很太非常十分特别相当比较更最"
    "个只条件位名台辆把些种类份块双对"
    "时候地方事情东西问题方面情况工作生活"
    "获取联系拨打发送提交确认通知参加完成"
    "是和的在了不到有人这那就也都可以"
)

# 百家姓正则：(复姓|单姓) + 1~2 个汉字 (排除常见非名字后缀)
_SURNAME_ALTS = "|".join(
    sorted([s for s in _COMMON_SURNAMES if len(s) == 2], key=len, reverse=True)
    + sorted([s for s in _COMMON_SURNAMES if len(s) == 1])
)
_NAME_PATTERN = re.compile(
    rf"(?P<name>(?:{_SURNAME_ALTS}){_CJK_CHAR}{{1,2}})"
)


class ChineseNameRecognizer(EntityRecognizer):
    """中文人名 Recognizer。

    主策略: 使用 spaCy 中文 NER (zh_core_web_trf / zh_core_web_sm) 识别 PERSON 实体，
    并通过百家姓前缀验证提升置信度。

    回退策略: 当 spaCy 中文模型不可用时，使用百家姓前缀 + 正则启发式匹配。

    Entity type: CN_NAME
    """

    ENTITIES = ["CN_NAME"]

    def __init__(
        self,
        nlp: Any | None = None,
        supported_language: str = "en",
        supported_entities: list[str] | None = None,
    ) -> None:
        """初始化中文人名识别器。

        Parameters
        ----------
        nlp : spacy.Language | None
            可选的 spaCy Language 实例 (应为中文模型)。
            如果未提供，尝试自动加载 zh_core_web_sm / zh_core_web_trf。
        supported_language : str
            支持的语言代码，默认 "en" (Presidio 默认语言码)。
        supported_entities : list[str] | None
            支持的实体类型列表，默认 ["CN_NAME"]。
        """
        super().__init__(
            supported_entities=supported_entities or self.ENTITIES,
            supported_language=supported_language,
            name="ChineseNameRecognizer",
        )
        self._nlp = nlp
        self._use_spacy = False
        self._context_words = _CONTEXT_WORDS

        # 尝试加载 spaCy 中文模型
        if self._nlp is not None:
            self._use_spacy = True
            logger.info("ChineseNameRecognizer: 使用注入的 spaCy 模型")
        else:
            self._nlp = self._try_load_spacy_model()

    def _try_load_spacy_model(self) -> Any | None:
        """尝试加载 spaCy 中文模型。"""
        try:
            import spacy

            # 优先尝试 trf 模型 (更准确)
            for model_name in ("zh_core_web_trf", "zh_core_web_sm"):
                try:
                    nlp = spacy.load(model_name)
                    self._use_spacy = True
                    logger.info(
                        "ChineseNameRecognizer: 加载 spaCy 模型 %s 成功", model_name
                    )
                    return nlp
                except OSError:
                    continue

            logger.info(
                "ChineseNameRecognizer: 未找到中文 spaCy 模型，回退到启发式匹配"
            )
        except ImportError:
            logger.warning(
                "ChineseNameRecognizer: spaCy 未安装，回退到启发式匹配"
            )

        return None

    # ------------------------------------------------------------------
    # Presidio EntityRecognizer 接口
    # ------------------------------------------------------------------

    def load(self) -> None:  # noqa: D102
        """加载资源 (Presidio 生命周期回调)。"""
        pass

    def analyze(
        self,
        text: str,
        entities: list[str] | None = None,
        nlp_artifacts: Any | None = None,
        **kwargs: Any,
    ) -> list[RecognizerResult]:
        """分析文本并返回检测到的中文人名。

        Parameters
        ----------
        text : str
            待分析文本。
        entities : list[str] | None
            需要检测的实体类型列表。
        nlp_artifacts : Any | None
            Presidio 传入的 NLP 分析结果 (此处不使用)。

        Returns
        -------
        list[RecognizerResult]
            检测到的人名实体列表。
        """
        if entities and "CN_NAME" not in entities:
            return []

        # 计算上下文增强分数
        context_boost = self._compute_context_boost(text)

        if self._use_spacy and self._nlp is not None:
            results = self._analyze_with_spacy(text, context_boost)
        else:
            results = self._analyze_with_heuristic(text, context_boost)

        return results

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _compute_context_boost(self, text: str) -> float:
        """根据上下文关键词计算置信度增强值。"""
        for word in self._context_words:
            if word in text:
                return 0.1
        return 0.0

    def _analyze_with_spacy(
        self, text: str, context_boost: float
    ) -> list[RecognizerResult]:
        """使用 spaCy NER 检测中文人名。"""
        results: list[RecognizerResult] = []
        doc = self._nlp(text)

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # 基础分数
                base_score = 0.7

                # 百家姓前缀增强
                if self._has_surname_prefix(ent.text):
                    base_score = 0.85

                # 上下文增强
                score = min(base_score + context_boost, 1.0)

                results.append(
                    RecognizerResult(
                        entity_type="CN_NAME",
                        start=ent.start_char,
                        end=ent.end_char,
                        score=score,
                    )
                )

        return results

    def _analyze_with_heuristic(
        self, text: str, context_boost: float
    ) -> list[RecognizerResult]:
        """使用百家姓前缀启发式匹配检测中文人名。"""
        results: list[RecognizerResult] = []
        used_spans: list[tuple[int, int]] = []

        for match in _NAME_PATTERN.finditer(text):
            full_name = match.group("name")
            start = match.start()
            end = match.end()

            # 尝试确定实际名字长度 (2 字或 3 字)
            # 判断姓氏长度
            surname_len = 1
            if len(full_name) >= 2 and full_name[:2] in _COMMON_SURNAMES:
                surname_len = 2

            # 给定名部分
            given_part = full_name[surname_len:]

            # 如果给定名是 2 个字，检查第 2 个字是否更像是非名字字符
            if len(given_part) == 2 and given_part[1] in _NON_NAME_CHARS:
                # 缩短为 surname + 1 char 名字
                full_name = full_name[: surname_len + 1]
                end = start + len(full_name)

            # 过滤过短的匹配 (单姓 + 1 字 = 2 字最短)
            if len(full_name) < 2:
                continue

            # 排除常见非人名词语
            if self._is_common_word(full_name):
                continue

            # 检查是否与已检测的区间重叠
            overlaps = False
            for used_start, used_end in used_spans:
                if start < used_end and end > used_start:
                    overlaps = True
                    break
            if overlaps:
                continue

            # 基础分数 (启发式匹配分数较低)
            base_score = 0.55

            # 3 字及以上名字更可能是人名
            if len(full_name) >= 3:
                base_score = 0.65

            # 复姓额外增强
            if surname_len == 2:
                base_score = 0.7

            # 上下文增强
            score = min(base_score + context_boost, 1.0)

            results.append(
                RecognizerResult(
                    entity_type="CN_NAME",
                    start=start,
                    end=end,
                    score=score,
                )
            )
            used_spans.append((start, end))

        return results

    def _has_surname_prefix(self, text: str) -> bool:
        """检查文本是否以常见姓氏开头。"""
        # 先检查复姓
        if len(text) >= 2 and text[:2] in _COMMON_SURNAMES:
            return True
        # 再检查单姓
        if len(text) >= 1 and text[0] in _COMMON_SURNAMES:
            return True
        return False

    def _is_common_word(self, text: str) -> bool:
        """简单过滤常见非人名词语。"""
        # 常见容易误匹配的词语
        _NON_NAME_WORDS = {
            "张口", "张开", "张贴", "王道", "王牌",
            "李子", "周末", "周围", "周期", "周年",
            "陈旧", "陈列", "陈述",
            "马上", "马路", "马力",
            "高中", "高度", "高级",
            "林业", "林区",
            "黄金", "黄色", "黄河",
            "杨柳", "杨树",
            "项目", "计划",
        }
        # 正则会按“姓氏 + 1~2 字”贪婪匹配，例如把“项目复盘”
        # 截成“项目复”。以前仅做完整相等判断，无法过滤这类长句误报。
        return any(text.startswith(word) for word in _NON_NAME_WORDS)
