from dataclasses import dataclass


@dataclass
class ExtractionTask:
    task_id: str
    title: str
    url: str
    status: str = "pending"
    content: str = ""
    extraction_method: str = ""
    error: str = ""
    fallback_content: str = ""
