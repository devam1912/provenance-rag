import os
import re
from typing import List, Dict, Any, Optional

class PolicyChunk:
    """Represents a single chunk of policy text with rich metadata for retrieval and grounding."""
    
    def __init__(
        self,
        chunk_id: str,
        text: str,
        source: str,
        headers: List[str],
        chunk_index: int
    ):
        self.chunk_id = chunk_id  # format: filename#chunk_idx
        self.text = text
        self.source = source
        self.headers = headers
        self.chunk_index = chunk_index

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the chunk to a dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": {
                "source": self.source,
                "headers": self.headers,
                "chunk_index": self.chunk_index
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyChunk":
        """Deserializes a dictionary back to a PolicyChunk."""
        meta = data.get("metadata", {})
        return cls(
            chunk_id=data["chunk_id"],
            text=data["text"],
            source=meta.get("source", ""),
            headers=meta.get("headers", []),
            chunk_index=meta.get("chunk_index", 0)
        )


class PolicyChunker:
    """Splits policy documents into logical chunks based on heading structures and size limits."""
    
    def __init__(self, max_chunk_size: int = 800, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def split_document(self, filepath: str) -> List[PolicyChunk]:
        """Reads a document and splits it into PolicyChunks based on headers and length."""
        filename = os.path.basename(filepath)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Split content by markdown header syntax
        lines = content.split("\n")
        
        chunks: List[PolicyChunk] = []
        current_headers = ["", "", ""]  # [H1, H2, H3]
        current_block: List[str] = []
        chunk_idx = 0

        for line in lines:
            # Check for headers
            h1_match = re.match(r"^#\s+(.+)$", line)
            h2_match = re.match(r"^##\s+(.+)$", line)
            h3_match = re.match(r"^###\s+(.+)$", line)

            is_header = any([h1_match, h2_match, h3_match])

            if is_header and current_block:
                # Flush the previous block before starting the new heading
                block_text = "\n".join(current_block).strip()
                if block_text:
                    sub_chunks = self._split_text_block(block_text, filename, current_headers.copy(), chunk_idx)
                    chunks.extend(sub_chunks)
                    chunk_idx += len(sub_chunks)
                current_block = []

            # Update current active headers
            if h1_match:
                current_headers[0] = h1_match.group(1).strip()
                current_headers[1] = ""
                current_headers[2] = ""
            elif h2_match:
                current_headers[1] = h2_match.group(1).strip()
                current_headers[2] = ""
            elif h3_match:
                current_headers[2] = h3_match.group(1).strip()

            current_block.append(line)

        # Flush the final block
        if current_block:
            block_text = "\n".join(current_block).strip()
            if block_text:
                sub_chunks = self._split_text_block(block_text, filename, current_headers.copy(), chunk_idx)
                chunks.extend(sub_chunks)

        return chunks

    def _split_text_block(
        self,
        text: str,
        filename: str,
        headers: List[str],
        start_chunk_idx: int
    ) -> List[PolicyChunk]:
        """Helper to split a single heading-grouped text block if it exceeds max size."""
        # Clean headers list
        active_headers = [h for h in headers if h]
        header_context = " > ".join(active_headers)

        # If block size fits, return it directly
        if len(text) <= self.max_chunk_size:
            # Prepend context to the text of the chunk so it is self-contained for vector retrieval
            full_text = f"Context: {header_context}\n\n{text}" if header_context else text
            chunk_id = f"{filename}#chunk_{start_chunk_idx}"
            return [PolicyChunk(chunk_id, full_text, filename, active_headers, start_chunk_idx)]

        # Otherwise, split by sentences or lines recursively
        chunks: List[PolicyChunk] = []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        current_sub_block: List[str] = []
        current_len = 0
        sub_idx = start_chunk_idx

        for sentence in sentences:
            if current_len + len(sentence) > self.max_chunk_size and current_sub_block:
                sub_text = " ".join(current_sub_block).strip()
                full_text = f"Context: {header_context}\n\n{sub_text}" if header_context else sub_text
                chunk_id = f"{filename}#chunk_{sub_idx}"
                chunks.append(PolicyChunk(chunk_id, full_text, filename, active_headers, sub_idx))
                sub_idx += 1
                
                # Keep overlap sentences (simple overlap logic)
                overlap_text = current_sub_block[-2:] if len(current_sub_block) >= 2 else current_sub_block[-1:]
                current_sub_block = list(overlap_text)
                current_len = sum(len(s) for s in current_sub_block)

            current_sub_block.append(sentence)
            current_len += len(sentence)

        if current_sub_block:
            sub_text = " ".join(current_sub_block).strip()
            full_text = f"Context: {header_context}\n\n{sub_text}" if header_context else sub_text
            chunk_id = f"{filename}#chunk_{sub_idx}"
            chunks.append(PolicyChunk(chunk_id, full_text, filename, active_headers, sub_idx))

        return chunks
