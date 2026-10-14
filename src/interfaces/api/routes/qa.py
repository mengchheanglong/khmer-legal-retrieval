import asyncio
import json
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.application.dtos import LegalQARequest, LegalQAResponse
from src.application.use_cases.answer_legal_qa import AnswerLegalQAUseCase
from src.interfaces.api.dependencies import get_qa_use_case

router = APIRouter(tags=["Legal Q&A"])


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        json_schema_extra={"example": "What are the essential elements required to form a valid contract in Cambodia?"},
    )
    top_k: int = Field(default=5, ge=1, le=20)
    law_filter: Optional[str] = Field(
        default=None,
        json_schema_extra={"example": "Civil Code 2007"},
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional model override (e.g. deepseek-chat, a3_xlmr)",
    )


@router.post("/qa", response_model=LegalQAResponse)
async def answer_question(
    payload: QuestionRequest,
    qa_use_case: AnswerLegalQAUseCase = Depends(get_qa_use_case),
) -> LegalQAResponse:
    """
    Generate a grounded legal answer citing specific articles from Cambodian law.
    """
    req = LegalQARequest(
        question=payload.question,
        top_k=payload.top_k,
        law_filter=payload.law_filter,
        model=payload.model,
    )
    return qa_use_case.execute(req)


@router.post("/qa/stream")
async def stream_answer_question(
    payload: QuestionRequest,
    qa_use_case: AnswerLegalQAUseCase = Depends(get_qa_use_case),
):
    """
    Stream real-time legal pipeline stages and final cited response via Server-Sent Events (SSE).
    Ensures the frontend UI visual state is in 100% sync with real backend execution.
    """
    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def on_stage(stage_code: str, message: str):
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "stage", "stage": stage_code, "message": message},
            )

        async def worker():
            try:
                req = LegalQARequest(
                    question=payload.question,
                    top_k=payload.top_k,
                    law_filter=payload.law_filter,
                    model=payload.model,
                )
                response = await asyncio.to_thread(qa_use_case.execute, req, on_stage)
                await queue.put({"type": "complete", "response": response.model_dump()})
            except Exception as e:
                await queue.put({"type": "error", "message": str(e)})

        task = asyncio.create_task(worker())

        while not task.done() or not queue.empty():
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            except asyncio.TimeoutError:
                continue

        await task

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
