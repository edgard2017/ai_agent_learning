"""创建由应用自己保存的多轮会话。"""

from pathlib import Path

from agents import SQLiteSession

from .config import PROJECT_ROOT


DEFAULT_SESSION_DB = PROJECT_ROOT / ".agent_data" / "conversations.db"


def build_session(
    session_id: str,
    db_path: str | Path = DEFAULT_SESSION_DB,
) -> SQLiteSession:
    """为一个聊天 ID 创建或重新打开 SQLite 会话。"""

    normalized_id = session_id.strip()
    if not normalized_id:
        raise ValueError("session_id 不能为空")

    database_path = Path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteSession(normalized_id, database_path)
