"""Conversation API routes – list, detail, turns for chat_turns."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import delete, func, select, distinct
from sqlalchemy.ext.asyncio import AsyncSession

from surogate.core.db.engine import get_session
from surogate.core.db.models.trace import ChatTurn
from surogate.server.auth.authentication import get_current_subject

router = APIRouter()


@router.get("/")
async def list_conversations(
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
    project_name: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
):
    """List conversations (grouped from chat_turns)."""
    # Aggregate per conversation_id
    base = select(
        ChatTurn.conversation_id,
        func.min(ChatTurn.created_at).label("started_at"),
        func.max(ChatTurn.created_at).label("last_turn_at"),
        func.count(ChatTurn.id).label("turn_count"),
        func.sum(ChatTurn.prompt_tokens).label("total_prompt_tokens"),
        func.sum(ChatTurn.completion_tokens).label("total_completion_tokens"),
        func.sum(ChatTurn.total_tokens).label("total_tokens"),
        func.avg(ChatTurn.latency_ms).label("avg_latency_ms"),
        func.max(ChatTurn.model).label("model"),
        func.max(ChatTurn.project_name).label("project_name"),
        func.max(ChatTurn.run_name).label("run_name"),
        func.bool_or(ChatTurn.compacted).label("has_compaction"),
    ).group_by(ChatTurn.conversation_id)

    if project_name:
        base = base.where(ChatTurn.project_name == project_name)
    if model:
        base = base.where(ChatTurn.model == model)

    # Wrap as subquery for ordering/pagination
    sub = base.subquery()

    count_stmt = select(func.count()).select_from(sub)
    total = (await session.execute(count_stmt)).scalar() or 0

    rows_stmt = (
        select(sub)
        .order_by(sub.c.last_turn_at.desc())
        .limit(limit)
        .offset(offset)
    )
    rows = (await session.execute(rows_stmt)).all()

    conversations = []
    for r in rows:
        conversations.append({
            "conversation_id": r.conversation_id,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "last_turn_at": r.last_turn_at.isoformat() if r.last_turn_at else None,
            "turn_count": r.turn_count,
            "total_prompt_tokens": r.total_prompt_tokens or 0,
            "total_completion_tokens": r.total_completion_tokens or 0,
            "total_tokens": r.total_tokens or 0,
            "avg_latency_ms": round(r.avg_latency_ms, 1) if r.avg_latency_ms else None,
            "model": r.model or "",
            "project_name": r.project_name or "",
            "run_name": r.run_name or "",
            "has_compaction": r.has_compaction or False,
        })

    return {"conversations": conversations, "total": total}


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """Get a single conversation with all its turns."""
    stmt = (
        select(ChatTurn)
        .where(ChatTurn.conversation_id == conversation_id)
        .order_by(ChatTurn.created_at.asc())
    )
    result = await session.execute(stmt)
    turns = result.scalars().all()

    if not turns:
        raise HTTPException(status_code=404, detail="Conversation not found")

    first = turns[0]
    last = turns[-1]

    total_prompt = sum(t.prompt_tokens or 0 for t in turns)
    total_completion = sum(t.completion_tokens or 0 for t in turns)
    latencies = [t.latency_ms for t in turns if t.latency_ms is not None]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else None

    # Extract messages from turns for the thread view
    messages = []
    for t in turns:
        req = t.request_body or {}
        turn_messages = req.get("messages", [])
        # Only add the last user message (the prompt for this turn)
        if turn_messages:
            last_msg = turn_messages[-1]
            if last_msg.get("role") == "user":
                messages.append({
                    "role": "user",
                    "content": last_msg.get("content", ""),
                    "timestamp": t.created_at.isoformat() if t.created_at else None,
                    "tokens": t.prompt_tokens,
                })

        # Add assistant response from response_body
        resp = t.response_body
        if resp:
            choices = resp.get("choices", [])
            if choices:
                assistant_msg = choices[0].get("message", {})
                content = assistant_msg.get("content", "")
                reasoning = assistant_msg.get("reasoning_content", "")
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning or None,
                    "timestamp": t.created_at.isoformat() if t.created_at else None,
                    "tokens": t.completion_tokens,
                    "latency": round(t.latency_ms) if t.latency_ms else None,
                })

    turn_details = []
    for t in turns:
        turn_details.append({
            "id": t.id,
            "parent_hash": t.parent_hash,
            "state_hash": t.state_hash,
            "tail_hash": t.tail_hash,
            "model": t.model,
            "is_streaming": t.is_streaming,
            "prompt_tokens": t.prompt_tokens,
            "completion_tokens": t.completion_tokens,
            "total_tokens": t.total_tokens,
            "latency_ms": t.latency_ms,
            "compacted": t.compacted,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        })

    # Extract system prompt from the first turn's request
    system_prompt = None
    first_req = first.request_body or {}
    for msg in first_req.get("messages", []):
        if msg.get("role") == "system":
            system_prompt = msg.get("content", "")
            break

    return {
        "conversation_id": conversation_id,
        "project_name": first.project_name,
        "run_name": first.run_name,
        "model": last.model,
        "system_prompt": system_prompt,
        "started_at": first.created_at.isoformat() if first.created_at else None,
        "last_turn_at": last.created_at.isoformat() if last.created_at else None,
        "turn_count": len(turns),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "avg_latency_ms": avg_latency,
        "has_compaction": any(t.compacted for t in turns),
        "messages": messages,
        "turns": turn_details,
    }


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    """Delete all turns belonging to a conversation."""
    stmt = delete(ChatTurn).where(ChatTurn.conversation_id == conversation_id)
    result = await session.execute(stmt)
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await session.commit()
    return {"status": "deleted", "turns_deleted": result.rowcount}
