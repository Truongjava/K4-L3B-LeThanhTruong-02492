from typing import Callable

from .store import EmbeddingStore

NO_CONTEXT_MESSAGE = (
    "Không tìm thấy thông tin nào trong cơ sở tri thức để trả lời câu hỏi này."
)


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def _source_label(self, metadata: dict) -> str:
        """Pick the most traceable label the metadata offers."""
        for key in ("source_url", "source", "doc_id"):
            value = metadata.get(key)
            if value:
                return str(value)
        return "không rõ nguồn"

    def _build_prompt(self, question: str, results: list[dict]) -> str:
        """Number each chunk so the answer can cite the one it came from."""
        blocks = [
            f"[{index}] (nguồn: {self._source_label(result['metadata'])})\n{result['content']}"
            for index, result in enumerate(results, start=1)
        ]
        return (
            "Bạn là hệ thống trả lời câu hỏi dựa trên tài liệu được cung cấp.\n"
            "Chỉ sử dụng thông tin trong phần NGỮ CẢNH dưới đây. Nếu ngữ cảnh không "
            "chứa câu trả lời, hãy nói rõ rằng không tìm thấy thông tin — không được "
            "suy đoán hay thêm kiến thức bên ngoài.\n"
            "Khi trả lời, hãy trích dẫn số hiệu của đoạn đã dùng, ví dụ [1].\n\n"
            f"NGỮ CẢNH:\n{chr(10).join(blocks)}\n\n"
            f"CÂU HỎI: {question}\n"
            "TRẢ LỜI:"
        )

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if not results:
            # Nothing retrieved: answering anyway would invite the model to make
            # something up, and there is no context to cite.
            return NO_CONTEXT_MESSAGE
        return self.llm_fn(self._build_prompt(question, results))