# main.py
"""
astrbot_plugin_fuxi_active —— 芙熙专属「主动搭话」插件

让芙熙在允许时段内，按【固定间隔 + 概率】主动来找哥哥说话。独立实现，
不依赖任何好感度 / 身体状态插件。

设计要点
--------
1. 事件缓存：用 `event_message_type(ALL, priority=-1)` 钩子缓存每个会话最近一次
   真实事件（`_last_events`）与平台信息（`_platform_cache`）。框架本身无法凭空
   发起消息，必须攒一个「模板事件」才能在搭话时合成。
2. 调度器：`asyncio` 后台循环，每 `interval_hours ± interval_float_hours` 巡检一轮。
   在允许时段内，对每个已知会话判定：
       沉默时长 >= min_silence_minutes  且  概率命中  且  未达每日上限
   → 触发一次搭话。每个会话每轮最多触发一次。
3. 发送：
   - 有缓存事件（aiocqhttp / 绝大多数情况）→ 合成一条 `AstrMessageEvent` 塞进框架
     事件队列，走【完整九阶段管线】：persona、分段、其它插件的 on_llm_request 全部生效。
   - 无缓存事件（如机器人重启后、哥哥还没发过消息）→ 直接调 `context.llm_generate`
     生成，再自行分段发送。
4. 配置：`_conf_schema.json` 提供默认值；聊天指令 `/搭话 ...` 写入运行时覆盖
   （存于插件数据目录的 state.json），二者都可调，无需重启。

作者：Tenet
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star

PLUGIN_NAME = "astrbot_plugin_fuxi_active"

# 不支持「主动消息」的平台（发送时会直接 raise），命中即跳过。
_NO_PROACTIVE_PLATFORMS = ("weixin_official_account",)

# 运行时状态文件的默认结构
_DEFAULT_STATE: Dict[str, Any] = {
    "last_user_ts": {},   # {session_id: epoch}  哥哥最后一次说话时间
    "last_talk_ts": {},   # {session_id: epoch}  上次主动搭话时间
    "daily_count": {},    # {session_id: {"date": "YYYY-MM-DD", "count": n}}
    "overrides": {},      # {config_key: value}  聊天指令写入的运行时覆盖
}

_DEFAULT_TALK_PROMPT = (
    "【系统指令 · 主动搭话】现在时间是 {current_time}（{weekday}，{time_of_day}），"
    "距离{user_name}上次说话已经过去 {last_interaction_ago}。"
    "请你以芙熙的身份，主动给{user_name}发一条消息，就像忽然想起他、想找他说话那样。"
    "可任选一种动机（不要说明你选了哪种）：纯粹想他了（最简短）；"
    "分享你自己生活里的具体小事（刚做了什么、看到什么、天气、游戏、花）；"
    "顺着你们最近聊过的话题续一句；按当前时间关心他（饭点、深夜、天气）；"
    "猜他此刻在做什么（只能推测，不能说『我看到』）。"
    "要求：自然、口语化，符合你平时发消息的语气和长度（日常一到两句），"
    "不要用任何标签、格式或旁白，不要让{user_name}察觉到这是系统让你发的。"
)


class FuxiActiveChat(Star):
    def __init__(self, context: Context, config: Optional[dict] = None):
        super().__init__(context)
        self.context = context
        self.framework_config: Dict[str, Any] = config or {}
        self._can_persist_config = hasattr(self.framework_config, "save_config")

        # ---- 数据目录与状态文件 ----
        self.data_dir = self._resolve_data_dir()
        self.state_file = self.data_dir / "state.json"
        self.state = self._load_state()

        # ---- 运行时缓存 ----
        self._last_events: Dict[str, AstrMessageEvent] = {}
        self._platform_cache: Dict[str, dict] = {}
        self._last_save = 0.0
        self._task: Optional[asyncio.Task] = None

        logger.info(
            f"[芙熙搭话] 插件加载 | 启用={self._cfg_bool('enabled', False)} "
            f"| 时段 {self._cfg_str('time_start', '09:00')}-{self._cfg_str('time_end', '23:30')} "
            f"| 间隔 {self._cfg_float('interval_hours', 2.0)}h±{self._cfg_float('interval_float_hours', 0.5)}h "
            f"| 概率 {self._cfg_int('probability', 15)}%"
        )

        # ---- 启动调度器 ----
        self._task = asyncio.create_task(self._scheduler())

    # ==================================================================
    # 生命周期
    # ==================================================================
    async def terminate(self):
        """插件卸载 / 停用时取消调度器并落盘。"""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        self._save_state(force=True)
        logger.info("[芙熙搭话] 调度器已停止。")

    # ==================================================================
    # 路径 / 配置 / 状态
    # ==================================================================
    def _resolve_data_dir(self) -> Path:
        try:
            from astrbot.api.star import StarTools  # type: ignore

            return Path(StarTools.get_data_dir())
        except Exception:
            try:
                cfg = self.context.get_config() or {}
                base = cfg.get("data", "./data")
                return Path(base) / "plugin_data" / PLUGIN_NAME
            except Exception:
                return Path(__file__).parent / "data"

    def _load_state(self) -> Dict[str, Any]:
        state = json.loads(json.dumps(_DEFAULT_STATE))  # 深拷贝默认结构
        try:
            if self.state_file.exists():
                raw = json.loads(self.state_file.read_text("utf-8"))
                if isinstance(raw, dict):
                    for key, default in _DEFAULT_STATE.items():
                        val = raw.get(key)
                        state[key] = val if isinstance(val, type(default)) else default
        except Exception as e:
            logger.warning(f"[芙熙搭话] 读取 state.json 失败，使用默认状态: {e}")
        return state

    def _save_state(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_save < 3:
            return  # 简单节流，避免每条消息都写盘
        self._last_save = now
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(
                json.dumps(self.state, ensure_ascii=False, indent=2), "utf-8"
            )
        except Exception as e:
            logger.warning(f"[芙熙搭话] 写入 state.json 失败: {e}")

    def _cfg(self, key: str, default: Any = None) -> Any:
        """运行时覆盖 > 框架配置 > 代码默认。"""
        ov = self.state.get("overrides", {})
        if key in ov and ov[key] is not None:
            return ov[key]
        if key in self.framework_config and self.framework_config[key] is not None:
            return self.framework_config[key]
        return default

    def _cfg_bool(self, key: str, default: bool) -> bool:
        val = self._cfg(key, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in ("1", "true", "yes", "on", "是", "开")
        return bool(val)

    def _cfg_int(self, key: str, default: int) -> int:
        try:
            return int(float(self._cfg(key, default)))
        except (TypeError, ValueError):
            return default

    def _cfg_float(self, key: str, default: float) -> float:
        try:
            return float(self._cfg(key, default))
        except (TypeError, ValueError):
            return default

    def _cfg_str(self, key: str, default: str) -> str:
        val = self._cfg(key, default)
        return str(val) if val is not None else default

    def _set_override(self, key: str, value: Any) -> None:
        """写入运行时配置。

        优先写回框架配置并落盘（这样 WebUI 与指令共用同一份配置，不会互相遮挡）；
        若当前环境不支持 `save_config()`，则退化为写入本地 state.json 的 overrides。
        """
        if self._can_persist_config:
            try:
                self.framework_config[key] = value
                self.framework_config.save_config()
                return
            except Exception as e:
                logger.warning(f"[芙熙搭话] 写回框架配置失败，改用本地覆盖: {e}")
        self.state.setdefault("overrides", {})[key] = value
        self._save_state(force=True)

    # ==================================================================
    # 事件缓存：为合成搭话事件提供模板
    # ==================================================================
    @filter.event_message_type(filter.EventMessageType.ALL, priority=100)
    async def _on_message(self, event: AstrMessageEvent) -> None:
        try:
            # 跳过自己合成的搭话事件，避免覆盖真实事件 / 误记哥哥时间
            if event.get_extra("_fuxi_active_synthetic"):
                return

            session_id = event.unified_msg_origin
            if not session_id:
                return

            self._last_events[session_id] = event
            platform = session_id.split(":")[0] if ":" in session_id else session_id
            if platform not in self._platform_cache and hasattr(event, "platform_meta"):
                self._platform_cache[platform] = {
                    "platform_meta": getattr(event, "platform_meta", None),
                    "self_id": getattr(getattr(event, "message_obj", None), "self_id", "") or "",
                }

            # 记录哥哥最后一次说话时间（仅真实消息）
            self.state.setdefault("last_user_ts", {})[session_id] = time.time()
            self._save_state()
        except Exception:
            pass

    # ==================================================================
    # 调度器
    # ==================================================================
    async def _scheduler(self) -> None:
        logger.info("[芙熙搭话] 调度器已启动。")
        first_round = True
        while True:
            try:
                if first_round:
                    await asyncio.sleep(60)  # 启动后 1 分钟先跑一轮，便于验证
                    first_round = False
                else:
                    base = max(0.1, self._cfg_float("interval_hours", 2.0))
                    fl = max(0.0, self._cfg_float("interval_float_hours", 0.5))
                    if fl > 0:
                        low = max(0.1, base - fl)
                        high = base + fl
                        wait_h = random.uniform(low, high)
                    else:
                        wait_h = base
                    logger.debug(f"[芙熙搭话] 本轮等待 {wait_h:.2f}h。")
                    await asyncio.sleep(wait_h * 3600)

                if not self._cfg_bool("enabled", False):
                    continue

                await self._run_round()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[芙熙搭话] 调度器出错: {e}")
                await asyncio.sleep(600)  # 出错后退避 10 分钟

    async def _run_round(self) -> None:
        now = datetime.now()
        if not self._in_time_window(now):
            logger.debug(f"[芙熙搭话] 当前 {now.strftime('%H:%M')} 不在允许时段，跳过本轮。")
            return

        sessions = list(self._last_events.keys())
        if not sessions:
            logger.debug("[芙熙搭话] 暂无已缓存会话（哥哥发过消息后才会出现），跳过本轮。")
            return

        random.shuffle(sessions)
        max_sessions = self._cfg_int("max_sessions_per_round", 1)
        triggered = 0

        for session_id in sessions:
            if max_sessions > 0 and triggered >= max_sessions:
                break
            if not self._session_allowed(session_id):
                continue
            if not self._silence_ok(session_id, now):
                continue
            if not self._under_daily_cap(session_id, now):
                continue

            prob = max(0, min(100, self._cfg_int("probability", 15)))
            if random.randint(1, 100) > prob:
                continue

            ok = await self._do_talk(session_id)
            if ok:
                triggered += 1
                self.state.setdefault("last_talk_ts", {})[session_id] = now.timestamp()
                self._bump_daily(session_id, now)
                self._save_state(force=True)

        if triggered:
            logger.info(f"[芙熙搭话] 本轮触发 {triggered} 个会话。")

    # ==================================================================
    # 判定辅助
    # ==================================================================
    def _in_time_window(self, now: datetime) -> bool:
        try:
            sh, sm = map(int, self._cfg_str("time_start", "09:00").split(":"))
            eh, em = map(int, self._cfg_str("time_end", "23:30").split(":"))
        except (ValueError, AttributeError):
            logger.warning("[芙熙搭话] 时段格式错误（需 HH:MM），跳过。")
            return False
        now_min = now.hour * 60 + now.minute
        return (sh * 60 + sm) <= now_min <= (eh * 60 + em)

    def _silence_ok(self, session_id: str, now: datetime) -> bool:
        need = self._cfg_int("min_silence_minutes", 30)
        if need <= 0:
            return True
        last_user = self.state.get("last_user_ts", {}).get(session_id)
        if not last_user:
            return True  # 从未记录（如刚重启），不拦
        return (now.timestamp() - float(last_user)) >= need * 60

    def _under_daily_cap(self, session_id: str, now: datetime) -> bool:
        cap = self._cfg_int("max_per_day", 3)
        if cap <= 0:
            return True
        today = now.strftime("%Y-%m-%d")
        entry = self.state.get("daily_count", {}).get(session_id)
        if not entry or entry.get("date") != today:
            return True
        return int(entry.get("count", 0)) < cap

    def _bump_daily(self, session_id: str, now: datetime) -> None:
        today = now.strftime("%Y-%m-%d")
        daily = self.state.setdefault("daily_count", {})
        entry = daily.get(session_id)
        if not entry or entry.get("date") != today:
            daily[session_id] = {"date": today, "count": 1}
        else:
            entry["count"] = int(entry.get("count", 0)) + 1

    def _session_allowed(self, session_id: str) -> bool:
        allowed = self._cfg("allowed_sessions", []) or []
        blocked = self._cfg("blocked_sessions", []) or []
        if self._match_session(session_id, blocked):
            return False
        if allowed and not self._match_session(session_id, allowed):
            return False
        return True

    @staticmethod
    def _match_session(session_id: str, patterns: List[str]) -> bool:
        """支持完整 UMO、平台前缀、以及尾缀（如纯 QQ 号）。"""
        sid = str(session_id)
        for p in patterns:
            p = str(p).strip()
            if not p:
                continue
            if sid == p or sid.startswith(p + ":") or sid.endswith(":" + p) or sid.split(":")[0] == p:
                return True
        return False

    @staticmethod
    def _adapter_name(last_event: Optional[AstrMessageEvent] = None,
                      platform_info: Optional[dict] = None) -> str:
        """取协议端适配器类型名（如 aiocqhttp）。

        会话 id 的前缀是平台实例的显示名（例如「芙熙（napcat）」），不是适配器类型；
        必须用 platform_meta.name 判断，否则自定义平台名会让 startswith 判断失效。
        """
        sources = (
            getattr(last_event, "platform_meta", None) if last_event is not None else None,
            (platform_info or {}).get("platform_meta"),
        )
        for pm in sources:
            name = getattr(pm, "name", "") if pm is not None else ""
            if name:
                return str(name).strip().lower()
        return ""

    @staticmethod
    def _platform_supports_proactive(session_id: str, adapter: str = "") -> bool:
        platform = (adapter or (session_id.split(":")[0] if ":" in session_id else session_id)).strip().lower()
        return bool(platform) and platform not in _NO_PROACTIVE_PLATFORMS

    @staticmethod
    def _time_of_day(now: datetime) -> str:
        h = now.hour
        if h < 5:
            return "深夜"
        if h < 8:
            return "清晨"
        if h < 11:
            return "上午"
        if h < 13:
            return "中午"
        if h < 17:
            return "下午"
        if h < 19:
            return "傍晚"
        if h < 23:
            return "晚上"
        return "深夜"

    @staticmethod
    def _humanize_ago(ts: Optional[float]) -> str:
        if not ts:
            return "有一段时间"
        delta = time.time() - float(ts)
        if delta < 60:
            return "刚刚"
        if delta < 3600:
            return f"{int(delta // 60)}分钟"
        if delta < 86400:
            return f"{int(delta // 3600)}小时"
        return f"{int(delta // 86400)}天"

    # ==================================================================
    # 搭话：构建提示词 + 发送
    # ==================================================================
    def _build_prompt(self, session_id: str) -> str:
        now = datetime.now()
        tpl = self._cfg_str("llm_prompt", _DEFAULT_TALK_PROMPT) or _DEFAULT_TALK_PROMPT
        weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]
        try:
            return tpl.format(
                current_time=now.strftime("%Y-%m-%d %H:%M:%S"),
                weekday=weekday,
                time_of_day=self._time_of_day(now),
                last_interaction_ago=self._humanize_ago(self.state.get("last_user_ts", {}).get(session_id)),
                user_name=self._cfg_str("user_name", "哥哥"),
            )
        except (KeyError, IndexError, ValueError) as e:
            logger.warning(f"[芙熙搭话] 提示词模板格式化失败（{e}），使用内置默认模板。")
            return _DEFAULT_TALK_PROMPT.format(
                current_time=now.strftime("%Y-%m-%d %H:%M:%S"),
                weekday=weekday,
                time_of_day=self._time_of_day(now),
                last_interaction_ago=self._humanize_ago(self.state.get("last_user_ts", {}).get(session_id)),
                user_name=self._cfg_str("user_name", "哥哥"),
            )

    async def _do_talk(self, session_id: str, template_event: Optional[AstrMessageEvent] = None) -> bool:
        """对指定会话执行一次搭话。返回是否成功发出。

        template_event：可选的真实事件。手动 /搭话 现在 时把当前命令事件传进来，
        当事件缓存为空（例如插件刚重载、哥哥还没发过消息）时用作合成模板，
        避免误落到绕过框架管线的"直接发送"。
        """
        prompt = self._build_prompt(session_id)
        platform = session_id.split(":")[0] if ":" in session_id else session_id
        last_event = self._last_events.get(session_id) or template_event
        platform_info = self._platform_cache.get(platform, {})
        adapter = self._adapter_name(last_event, platform_info)

        if not self._platform_supports_proactive(session_id, adapter):
            logger.warning(f"[芙熙搭话] 平台不支持主动消息，跳过 {session_id}。")
            return False

        # 有模板事件、且适配器类型兼容（未知时也尝试）→ 合成事件走完整管线
        uses_synthetic = (
            last_event is not None and (adapter == "" or adapter.startswith("aiocqhttp"))
        )
        if uses_synthetic:
            try:
                synth = self._make_synthetic_event(session_id, prompt, last_event, platform_info)
                self.context.get_event_queue().put_nowait(synth)
                logger.info(f"[芙熙搭话] 触发（合成事件管线）→ {session_id} | adapter={adapter or '未知'}")
                return True
            except Exception as e:
                logger.warning(f"[芙熙搭话] 合成事件失败，回退直接发送: {e}")

        # 回退：直接调 LLM + 发送
        logger.warning(f"[芙熙搭话] 回退直接发送（无模板事件或适配器不兼容）→ {session_id} | adapter={adapter or '未知'}")
        return await self._send_direct(session_id, prompt)

    def _make_synthetic_event(
        self,
        session_id: str,
        prompt: str,
        last_event: Optional[AstrMessageEvent],
        platform_info: dict,
    ) -> AstrMessageEvent:
        """构造一条「伪装成用户发来」的合成事件，塞进框架队列走完整管线。"""
        from astrbot.core.message.components import Plain
        from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
        from astrbot.core.platform.astr_message_event import AstrMessageEvent as _Event
        from astrbot.core.platform.message_type import MessageType

        target_uid = ""
        if last_event is not None:
            try:
                target_uid = str(last_event.get_sender_id())
            except Exception:
                target_uid = ""

        msg = AstrBotMessage()
        if last_event is not None:
            obj = last_event.message_obj
            msg.type = getattr(obj, "type", MessageType.FRIEND_MESSAGE)
            msg.self_id = getattr(obj, "self_id", "")
            group_id = getattr(obj, "group_id", "")
            platform_meta = last_event.platform_meta
        else:
            msg.type = MessageType.GROUP_MESSAGE if "GroupMessage" in session_id else MessageType.FRIEND_MESSAGE
            msg.self_id = platform_info.get("self_id", "")
            group_id = session_id.split(":")[2] if ("GroupMessage" in session_id and session_id.count(":") >= 2) else ""
            platform_meta = platform_info.get("platform_meta")

        msg.session_id = session_id
        msg.message_id = f"fuxi_active_{int(time.time())}"
        msg.sender = MessageMember(user_id=target_uid or session_id, nickname=target_uid or session_id)
        msg.message = [Plain(text=prompt)]
        msg.message_str = prompt
        msg.raw_message = None
        msg.timestamp = int(time.time())
        if group_id:
            msg.group_id = group_id

        event = _Event(
            message_str=prompt,
            message_obj=msg,
            platform_meta=platform_meta,
            session_id=session_id,
        )
        event.is_at_or_wake_command = True   # 强制当作需要回复
        event.call_llm = False               # 允许触发 LLM
        event.set_extra("_fuxi_active_synthetic", True)
        return event

    async def _send_direct(self, session_id: str, prompt: str) -> bool:
        """无缓存事件时：直接调 LLM 生成搭话内容并分段发送。"""
        try:
            provider_id = None
            if hasattr(self.context, "get_current_chat_provider_id"):
                try:
                    provider_id = await self.context.get_current_chat_provider_id(umo=session_id)
                except Exception:
                    provider_id = None
            if not provider_id and hasattr(self.context, "get_using_provider_async"):
                prov = await self.context.get_using_provider_async(session_id)
                provider_id = getattr(prov, "id", None) if prov else None
            if not provider_id:
                logger.warning(f"[芙熙搭话] 会话 {session_id} 无可用 LLM provider，跳过。")
                return False

            sys_prompt = await self._get_persona_prompt(session_id)
            contexts = await self._get_history_contexts(session_id)

            resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=sys_prompt,
                contexts=contexts,
            )
            text = (resp.completion_text or "").strip() if resp else ""
            if not text:
                logger.warning(f"[芙熙搭话] LLM 未生成内容（{session_id}）。")
                return False

            await self._send_segmented(session_id, text)
            await self._write_history(session_id, prompt, text)
            logger.info(f"[芙熙搭话] 触发（直接发送）→ {session_id} | {len(text)} 字")
            return True
        except Exception as e:
            logger.error(f"[芙熙搭话] 直接发送失败（{session_id}）: {e}")
            return False

    async def _get_persona_prompt(self, session_id: str) -> str:
        try:
            pm = getattr(self.context, "persona_manager", None)
            if pm and hasattr(pm, "get_default_persona_v3"):
                persona = await pm.get_default_persona_v3(umo=session_id)
                if isinstance(persona, dict):
                    return persona.get("prompt", "") or ""
        except Exception as e:
            logger.debug(f"[芙熙搭话] 获取 persona 失败: {e}")
        return ""

    async def _get_history_contexts(self, session_id: str) -> list:
        """读取会话当前对话历史，转为 llm_generate 的 contexts 参数。失败返回 []。"""
        try:
            from astrbot.core.agent.message import bind_checkpoint_messages, is_checkpoint_message  # type: ignore

            conv_mgr = self.context.conversation_manager
            cid = await conv_mgr.get_curr_conversation_id(session_id)
            if not cid:
                return []
            conversation = await conv_mgr.get_conversation(session_id, cid)
            if not conversation or not conversation.history:
                return []
            history = json.loads(conversation.history)
            if not isinstance(history, list):
                return []
            history = [m for m in history if not is_checkpoint_message(m)]
            history = history[-40:]  # 最多 20 轮
            return bind_checkpoint_messages(history)
        except Exception as e:
            logger.debug(f"[芙熙搭话] 读取历史失败，将不带上下文: {e}")
            return []

    async def _send_segmented(self, session_id: str, text: str) -> None:
        """按自然句边界分段发送，段间加延迟，模拟真人打字。"""
        for seg in self._segment(text):
            seg = seg.strip()
            if not seg:
                continue
            try:
                await self.context.send_message(session_id, MessageChain().message(seg))
            except Exception as e:
                logger.warning(f"[芙熙搭话] 发送失败: {e}")
            await asyncio.sleep(min(0.4 + len(seg) * 0.08, 2.5))

    @staticmethod
    def _segment(text: str, soft_limit: int = 120) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        parts = re.split(r"(?<=[。！？!?\n])", text)
        merged: List[str] = []
        buf = ""
        for p in parts:
            if not p:
                continue
            if len(buf) + len(p) <= soft_limit:
                buf += p
            else:
                if buf:
                    merged.append(buf)
                buf = p
        if buf:
            merged.append(buf)
        return merged

    async def _write_history(self, session_id: str, prompt: str, reply: str) -> None:
        """把搭话写进对话历史，让芙熙后续能记得自己找过哥哥。"""
        try:
            conv_mgr = self.context.conversation_manager
            cid = await conv_mgr.get_curr_conversation_id(session_id)
            if not cid:
                try:
                    cid = await conv_mgr.new_conversation(unified_msg_origin=session_id)
                except Exception:
                    cid = None
            if not cid:
                return
            user_msg = {"role": "user", "content": [{"type": "text", "text": f"[系统主动搭话触发] {prompt}"}]}
            asst_msg = {"role": "assistant", "content": [{"type": "text", "text": reply}]}
            await conv_mgr.add_message_pair(cid=cid, user_message=user_msg, assistant_message=asst_msg)
        except Exception as e:
            logger.debug(f"[芙熙搭话] 写入历史失败: {e}")

    # ==================================================================
    # 聊天指令
    # ==================================================================
    def _is_admin(self, event: AstrMessageEvent) -> bool:
        try:
            cfg = self.context.get_config() or {}
            admins = [str(a) for a in (cfg.get("admins_id", []) or [])]
            if str(event.get_sender_id()) in admins:
                return True
        except Exception:
            pass
        try:
            return bool(event.is_admin())
        except Exception:
            return False

    @filter.command("搭话", alias={"主动搭话", "芙熙搭话"})
    async def cmd_talk(self, event: AstrMessageEvent, action: str = "", arg1: str = "", arg2: str = ""):
        """主动搭话指令：/搭话 [状态|开|关|现在|概率 N|间隔 H|时段 HH:MM HH:MM|沉默 N|上限 N]"""
        if not self._is_admin(event):
            yield event.plain_result("这个指令只给哥哥用哦～")
            return

        action = (action or "").strip()

        if action in ("", "帮助", "help", "状态", "status"):
            yield event.plain_result(self._status_text(event.unified_msg_origin))
            return

        if action in ("开", "开启", "on", "启用"):
            self._set_override("enabled", True)
            yield event.plain_result("好，芙熙会主动来找你了。")
            return

        if action in ("关", "关闭", "off", "停用"):
            self._set_override("enabled", False)
            yield event.plain_result("嗯，那芙熙先不主动打扰你了。")
            return

        if action in ("现在", "now", "立即"):
            # 先用当前命令事件刷新缓存：即便插件刚重载、缓存为空，也能走合成事件管线
            sid = event.unified_msg_origin
            self._last_events[sid] = event
            plat = sid.split(":")[0] if ":" in sid else sid
            if hasattr(event, "platform_meta"):
                self._platform_cache.setdefault(plat, {
                    "platform_meta": getattr(event, "platform_meta", None),
                    "self_id": getattr(getattr(event, "message_obj", None), "self_id", "") or "",
                })
            ok = await self._do_talk(sid, template_event=event)
            if ok:
                self.state.setdefault("last_talk_ts", {})[sid] = time.time()
                self._bump_daily(sid, datetime.now())
                self._save_state(force=True)
                yield event.plain_result("（好啦，已经找过你了）")
            else:
                yield event.plain_result("这次没发出去，看看日志吧。")
            return

        if action in ("概率", "probability"):
            try:
                n = int(arg1)
            except (TypeError, ValueError):
                yield event.plain_result("用法：/搭话 概率 15（0-100）")
                return
            n = max(0, min(100, n))
            self._set_override("probability", n)
            yield event.plain_result(f"每轮触发概率调成 {n}% 了。")
            return

        if action in ("间隔", "interval"):
            try:
                h = float(arg1)
            except (TypeError, ValueError):
                yield event.plain_result("用法：/搭话 间隔 2（小时，可用小数）")
                return
            h = max(0.1, h)
            self._set_override("interval_hours", h)
            yield event.plain_result(f"巡检间隔调成 {h} 小时了（下次巡检生效）。")
            return

        if action == "时段":
            if not self._valid_hm(arg1) or not self._valid_hm(arg2):
                yield event.plain_result("用法：/搭话 时段 09:00 23:30")
                return
            self._set_override("time_start", arg1)
            self._set_override("time_end", arg2)
            yield event.plain_result(f"允许搭话时段调成 {arg1}–{arg2} 了。")
            return

        if action in ("沉默", "silence"):
            try:
                m = int(arg1)
            except (TypeError, ValueError):
                yield event.plain_result("用法：/搭话 沉默 30（分钟，0=不限制）")
                return
            self._set_override("min_silence_minutes", max(0, m))
            yield event.plain_result(f"最少沉默时长调成 {max(0, m)} 分钟了。")
            return

        if action in ("上限", "cap"):
            try:
                m = int(arg1)
            except (TypeError, ValueError):
                yield event.plain_result("用法：/搭话 上限 3（0=不限制）")
                return
            self._set_override("max_per_day", max(0, m))
            yield event.plain_result(f"每日上限调成 {max(0, m)} 次了。")
            return

        yield event.plain_result("没听懂的指令，发 /搭话 看帮助～")

    @staticmethod
    def _valid_hm(s: str) -> bool:
        try:
            h, m = map(int, str(s).split(":"))
            return 0 <= h <= 23 and 0 <= m <= 59
        except (ValueError, AttributeError):
            return False

    def _status_text(self, session_id: str) -> str:
        en = "开" if self._cfg_bool("enabled", False) else "关"
        today = datetime.now().strftime("%Y-%m-%d")
        entry = self.state.get("daily_count", {}).get(session_id, {})
        used = entry.get("count", 0) if entry.get("date") == today else 0
        last_talk = self.state.get("last_talk_ts", {}).get(session_id)
        last_talk_str = datetime.fromtimestamp(last_talk).strftime("%m-%d %H:%M") if last_talk else "还没搭过"
        return (
            "【芙熙搭话 · 状态】\n"
            f"开关：{en}\n"
            f"允许时段：{self._cfg_str('time_start', '09:00')} – {self._cfg_str('time_end', '23:30')}\n"
            f"巡检间隔：{self._cfg_float('interval_hours', 2.0)}h ± {self._cfg_float('interval_float_hours', 0.5)}h\n"
            f"触发概率：{self._cfg_int('probability', 15)}%\n"
            f"最少沉默：{self._cfg_int('min_silence_minutes', 30)} 分钟\n"
            f"每日上限：{self._cfg_int('max_per_day', 3)}（今日已用 {used}）\n"
            f"上次搭话：{last_talk_str}\n"
            "指令：/搭话 开|关|现在|概率 N|间隔 H|时段 HH:MM HH:MM|沉默 N|上限 N"
        )
