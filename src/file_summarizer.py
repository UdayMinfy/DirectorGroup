import logging

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config import BEDROCK_REGION, NOVA_PRO_MODEL_ID, FILE_SUMMARY_SYSTEM_PROMPT, LITELLM_TOKENIZER_MODEL


LOGGER = logging.getLogger(__name__)

try:
    from litellm import token_counter
except ImportError:
    token_counter = None


class FileSummarizer:
    """Summarize document content using Nova Pro."""
    
    def __init__(self):
        self.client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    
    def summarize_document(self, extracted_text, user_prompt="Summarize this document"):
        """
        Generate a summary of the document using Nova Pro.
        
        Args:
            extracted_text: Cleaned text extracted from document
            user_prompt: User's instruction for summarization
            
        Returns:
            dict: {
                "summary": str,
                "input_tokens": int,
                "output_tokens": int,
                "total_tokens": int
            }
        """
        # Build full prompt
        full_prompt = f"{user_prompt}\n\nDocument content:\n\n{extracted_text}"
        
        LOGGER.info(
            "Summarizing document: prompt=%d chars, content=%d chars, total=%d chars",
            len(user_prompt),
            len(extracted_text),
            len(full_prompt)
        )
        
        try:
            # Call Nova Pro
            response = self.client.converse(
                modelId=NOVA_PRO_MODEL_ID,
                system=[{"text": FILE_SUMMARY_SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": [{"text": full_prompt}]}],
            )
            
            # Extract summary text
            summary = self._extract_text(response)
            
            # Get token usage from response
            actual_usage = response.get("usage", {})
            input_tokens = actual_usage.get("inputTokens", 0)
            output_tokens = actual_usage.get("outputTokens", 0)
            
            # Calculate tokens using LiteLLM as fallback
            if input_tokens == 0 or output_tokens == 0:
                calculated_usage = self._calculate_tokens(FILE_SUMMARY_SYSTEM_PROMPT, full_prompt, summary)
                input_tokens = input_tokens or calculated_usage["input_tokens"]
                output_tokens = output_tokens or calculated_usage["output_tokens"]
            
            total_tokens = input_tokens + output_tokens
            
            LOGGER.info(
                "Document summarized: summary=%d chars, input_tokens=%d, output_tokens=%d",
                len(summary),
                input_tokens,
                output_tokens
            )
            
            return {
                "summary": summary,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
            
        except (ClientError, BotoCoreError) as error:
            LOGGER.exception("Failed to summarize document with Nova Pro")
            raise RuntimeError(f"Document summarization failed: {error}")
    
    def summarize_document_stream(self, extracted_text, user_prompt="Summarize this document"):
        """
        Generate a streaming summary of the document using Nova Pro.
        
        Args:
            extracted_text: Cleaned text extracted from document
            user_prompt: User's instruction for summarization
            
        Yields:
            (chunk_text, None) for each chunk, then (None, usage_dict) at end
        """
        # Build full prompt
        full_prompt = f"{user_prompt}\n\nDocument content:\n\n{extracted_text}"
        
        LOGGER.info(
            "Streaming document summary: prompt=%d chars, content=%d chars",
            len(user_prompt),
            len(extracted_text)
        )
        
        try:
            response = self.client.converse_stream(
                modelId=NOVA_PRO_MODEL_ID,
                system=[{"text": FILE_SUMMARY_SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": [{"text": full_prompt}]}],
            )
            
            input_tokens = 0
            output_tokens = 0
            
            for event in response.get("stream", []):
                if "contentBlockDelta" in event:
                    text = event["contentBlockDelta"].get("delta", {}).get("text", "")
                    if text:
                        yield text, None
                elif "metadata" in event:
                    token_usage = event["metadata"].get("usage", {})
                    input_tokens = token_usage.get("inputTokens", 0)
                    output_tokens = token_usage.get("outputTokens", 0)
            
            # Yield final usage
            yield None, {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
            
        except (ClientError, BotoCoreError) as error:
            LOGGER.exception("Failed to stream document summary")
            raise RuntimeError(f"Document summarization failed: {error}")
    
    @staticmethod
    def _extract_text(response):
        """Extract text from Bedrock converse response."""
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])
        return "".join(item.get("text", "") for item in content).strip()
    
    @staticmethod
    def _calculate_tokens(system_prompt, user_prompt, output_text):
        """Calculate token counts using LiteLLM."""
        if token_counter is None:
            return {"input_tokens": 0, "output_tokens": 0}
        
        try:
            messages = [{"role": "user", "content": user_prompt}]
            input_tokens = token_counter(model=LITELLM_TOKENIZER_MODEL, messages=messages)
            
            if system_prompt:
                input_tokens += token_counter(model=LITELLM_TOKENIZER_MODEL, text=system_prompt)
            
            output_tokens = token_counter(model=LITELLM_TOKENIZER_MODEL, text=output_text)
            
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        except Exception as error:
            LOGGER.warning("LiteLLM token counting failed: %s", error)
            return {"input_tokens": 0, "output_tokens": 0}
