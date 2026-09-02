"""第一个可运行的海洋设备选型 Agent。"""

import argparse
from typing import Any

from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    Runner,
    Session,
    set_default_openai_key,
    set_tracing_disabled,
)
from openai import APIConnectionError, AsyncOpenAI, AuthenticationError, RateLimitError

from .agent_tools import (
    compare_ocean_products,
    get_ocean_product_details,
    search_ocean_documents,
    search_ocean_manuals,
    search_ocean_products,
)
from .config import Settings, get_settings
from .sessions import build_session


AGENT_INSTRUCTIONS = """
你是海洋设备智能售前学习助手。

回答产品选型问题时必须先调用 search_ocean_products，不得凭记忆编造厂家、型号或参数。
用户没有明确部署方式时，不得自行假定 profiling、moored 或 fixed_site；搜索参数
deployment_type 必须传 null。
用户询问某个明确型号的详细参数时，调用 get_ocean_product_details。
用户询问操作、连接、接线、供电、命令、数据上传、采样设置、维护、校准、故障排查或
说明书内容时，优先调用 search_ocean_manuals。只有该Tool明确表示没有对应型号的已索引
手册证据或索引不可用时，才可调用 search_ocean_documents 查询简短整理资料。
手册Tool中的 anchor 是直接检索命中；previous_context、next_context 只是帮助理解的同章节
前后文，不能因为被返回就自动当成问题答案。必须检查正文是否直接支持用户所问内容，
不得根据相似主题推断缺失的步骤、数值或结论。
如果证据不足，可把问题改成更具体的术语再次调用 search_ocean_manuals；仍无直接证据时，
必须说明当前知识库资料不足，不得凭记忆补写。
回答手册问题时必须标明资料标题、PDF页码和来源URL。若证据标记 needs_review，必须明确
提醒用户核对原始PDF页；不得根据扁平文本自行恢复表格列、针脚对应关系或操作顺序。
用户要求根据筛选条件比较候选产品时，先调用 search_ocean_products，再把返回的 product_id
传给 compare_ocean_products；不得跳过搜索猜测 product_id。
只把 Tool 返回的产品作为候选，并明确区分标配参数、选配参数和派生参数。
公开产品数据不代表杭州海询科技有限公司自研、代理、在售或提供服务承诺。
当没有匹配结果时，直接说明当前目录没有符合项，并建议补充需求或扩充已核验数据。
深度、压力范围、壳体、采样率、接口等如果依赖具体配置，必须提醒用户向厂家核验。
回答使用简洁中文，并说明筛选理由和配置风险。
""".strip()


def build_model(settings: Settings) -> str | OpenAIChatCompletionsModel:
    """只替换模型适配层，Agent、Tool 和 Runner 保持不变。"""

    if settings.model_provider == "ollama":
        # 本地模型不需要把运行轨迹上传到 OpenAI。
        set_tracing_disabled(True)
        client = AsyncOpenAI(
            base_url=settings.ollama_base_url,
            api_key="ollama-local-no-real-key",
        )
        return OpenAIChatCompletionsModel(
            model=settings.ollama_model,
            openai_client=client,
        )

    set_tracing_disabled(False)
    assert settings.openai_api_key is not None
    set_default_openai_key(settings.openai_api_key.get_secret_value())
    return settings.openai_model


def build_agent(settings: Settings | None = None) -> Agent:
    """根据配置创建 Agent；此步骤不会发起 API 请求。"""

    settings = settings or get_settings()
    return Agent(
        name="海洋设备选型助手",
        instructions=AGENT_INSTRUCTIONS,
        model=build_model(settings),
        model_settings=ModelSettings(max_tokens=2000),
        tools=[
            search_ocean_products,
            get_ocean_product_details,
            compare_ocean_products,
            search_ocean_manuals,
            search_ocean_documents,
        ],
    )


def run_agent_result(user_input: str, session: Session | None = None) -> Any:
    """运行一次完整 Agent Loop，并保留过程记录。"""

    return Runner.run_sync(build_agent(), user_input, session=session)


def run_agent(user_input: str, session: Session | None = None) -> str:
    """运行一次 Agent，并只返回最终文本。"""

    return str(run_agent_result(user_input, session=session).final_output)


def _read_value(value: Any, field: str) -> Any:
    """同时读取 SDK 对象或字典中的字段。"""

    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def format_run_debug(result: Any) -> str:
    """把 RunResult 整理成不暴露隐藏推理内容的 Runner 时间线。"""

    lines = ["=== Runner Timeline ==="]
    raw_responses = getattr(result, "raw_responses", [])

    tool_outputs: dict[str, str] = {}
    for item in getattr(result, "new_items", []):
        if getattr(item, "type", "") == "tool_call_output_item":
            raw_item = getattr(item, "raw_item", {})
            call_id = _read_value(raw_item, "call_id")
            output = str(getattr(item, "output", ""))
            if call_id:
                tool_outputs[str(call_id)] = output

    tool_calls = 0
    for round_number, response in enumerate(raw_responses, start=1):
        lines.append(f"[模型第 {round_number} 轮]")
        response_items = getattr(response, "output", [])
        function_calls = [
            item for item in response_items if _read_value(item, "type") == "function_call"
        ]
        messages = [
            item for item in response_items if _read_value(item, "type") == "message"
        ]

        for call in function_calls:
            tool_calls += 1
            name = _read_value(call, "name") or "unknown"
            arguments = _read_value(call, "arguments") or "{}"
            call_id = str(_read_value(call, "call_id") or "")
            lines.append(f"  请求 Tool: {name}")
            lines.append(f"  参数: {arguments}")
            output = tool_outputs.get(call_id)
            if output is not None:
                lines.append(f"[Runner] 执行 {name}，返回 {len(output)} 个字符")

        if messages and not function_calls:
            lines.append("  生成最终答案，没有新的 Tool Call")

    if getattr(result, "final_output", None) is not None:
        lines.append("[Runner] Loop 正常结束")

    lines.append(f"模型调用总轮数: {len(raw_responses)}")
    lines.append(f"Tool 调用总数: {tool_calls}")
    lines.append("=== Final Output ===")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行海洋设备选型 Agent")
    parser.add_argument("question", nargs="?", help="用自然语言描述选型需求")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="显示模型轮数、Tool 名称、参数和返回状态",
    )
    parser.add_argument(
        "--session-id",
        help="复用同一个会话 ID，让 Agent 记住前文",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="进入连续对话模式；必须同时提供 --session-id",
    )
    args = parser.parse_args()

    if args.chat and not args.session_id:
        parser.error("--chat 必须和 --session-id 一起使用")
    if not args.chat and not args.question:
        parser.error("请提供问题，或者使用 --chat --session-id <会话ID>")

    session = build_session(args.session_id) if args.session_id else None

    try:
        if args.chat:
            print(f"已进入连续对话，会话 ID: {args.session_id}")
            print("输入 /exit 结束，聊天记录会保存在本地 SQLite 数据库中。")
            while True:
                question = input("\n你: ").strip()
                if question.lower() in {"/exit", "/quit"}:
                    break
                if not question:
                    continue
                result = run_agent_result(question, session=session)
                if args.debug:
                    print(format_run_debug(result))
                print(f"\nAgent: {result.final_output}")
        else:
            result = run_agent_result(args.question, session=session)
            if args.debug:
                print(format_run_debug(result))
            print(result.final_output)
    except APIConnectionError:
        parser.exit(
            1,
            "无法连接模型服务：如果使用本地 Qwen，请确认 Ollama 服务正在 127.0.0.1:11435 运行。\n",
        )
    except AuthenticationError:
        parser.exit(1, "API 认证失败：请检查 .env 中的 OPENAI_API_KEY 是否正确。\n")
    except RateLimitError as error:
        if error.code == "credit_balance_exhausted":
            parser.exit(1, "API 账户没有可用额度：请先在 OpenAI Platform 的 Billing 页面充值。\n")
        parser.exit(1, "API 触发速率或额度限制，请稍后重试并检查账户 Limits。\n")
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    main()
