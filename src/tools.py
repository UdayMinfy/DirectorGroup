import ast
import operator
import re

from bedrock_client import BedrockSummarizer
from browser_extractor import BrowserExtractor
from intent_router import BedrockIntentRouter, BedrockKnowledgeResponder
from web_search import WebSearcher


_VAGUE_PATTERNS = [
    re.compile(r'^tell me (more )?about (it|this|that)[.?!]?$', re.IGNORECASE),
    re.compile(r'^explain (this|that|it)[.?!]?$', re.IGNORECASE),
    re.compile(r'^give me (more )?(details?|info(rmation)?)[.?!]?$', re.IGNORECASE),
    re.compile(r'^more (details?|info(rmation)?)[.?!]?$', re.IGNORECASE),
    re.compile(r'^elaborate[.?!]?$', re.IGNORECASE),
    re.compile(r'^what (is|are) (it|this|that)[.?!]?$', re.IGNORECASE),
    re.compile(r'^can you (explain|tell me|describe) (it|this|that)[.?!]?$', re.IGNORECASE),
    re.compile(r'^(what|how|why|when|where)[.?!]?$', re.IGNORECASE),
    re.compile(r'^go on[.?!]?$', re.IGNORECASE),
    re.compile(r'^(and|so)[.?!]?$', re.IGNORECASE),
]


def is_vague_prompt(prompt: str) -> bool:
    cleaned = (prompt or "").strip()
    if not cleaned:
        return True
    for pattern in _VAGUE_PATTERNS:
        if pattern.match(cleaned):
            return True
    return False


def evaluate_math_expression(expression):
    """
    Safely evaluate a mathematical expression using Python AST.
    Supports: +, -, *, /, //, %, **, parentheses, unary + and -
    """
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')

        # Define allowed operations
        allowed_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.UAdd: operator.pos,
            ast.USub: operator.neg,
        }

        def eval_node(node):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError("Only numbers allowed")
            elif isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                op = allowed_ops.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported binary operator: {type(node.op)}")
                return op(left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = eval_node(node.operand)
                op = allowed_ops.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported unary operator: {type(node.op)}")
                return op(operand)
            else:
                raise ValueError(f"Unsupported AST node: {type(node)}")

        return eval_node(tree.body)

    except (SyntaxError, ValueError, TypeError, ZeroDivisionError):
        return None


def is_math_expression(prompt):
    """
    Check if the prompt is a simple arithmetic expression.
    """
    # Clean the prompt
    prompt = prompt.strip().lower()

    # Remove common prefixes like "what is", "calculate", etc.
    prefixes = [
        r'^what is\s+',
        r'^calculate\s+',
        r'^compute\s+',
        r'^solve\s+',
        r'^eval\s+',
        r'^evaluate\s+',
    ]
    for prefix in prefixes:
        prompt = re.sub(prefix, '', prompt)

    prompt = prompt.strip()

    # Remove trailing punctuation
    prompt = re.sub(r'[?!.]+$', '', prompt).strip()

    # Try to evaluate as math expression
    result = evaluate_math_expression(prompt)
    return result is not None, result


def clean_response(text: str) -> str:
    """
    Clean the response text by removing markdown formatting and excessive whitespace.
    """
    import re

    # Remove markdown headings
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Remove markdown bold markers
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)

    # Collapse excessive blank lines (3 or more newlines to 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Trim trailing whitespace
    text = text.rstrip()

    return text


class SearchWebTool:
    def __init__(self):
        self.searcher = WebSearcher()

    def run(self, query, limit):
        return self.searcher.search(query, limit=limit)


class BrowseExtractTool:
    def __init__(self):
        self.extractor = BrowserExtractor()

    def run(self, url):
        return self.extractor.extract(url)

    def run_with_session(self, url, session):
        return self.extractor.extract_with_session(url, session)


class SummarizeTool:
    def __init__(self):
        self.summarizer = BedrockSummarizer()

    def run(self, prompt, sources):
        return self.summarizer.summarize(prompt, sources)


class IntentTool:
    def __init__(self):
        self.router = BedrockIntentRouter()

    def run(self, prompt):
        return self.router.classify(prompt)

    def should_read_history(self, prompt):
        return self.router.should_read_history(prompt)


class KnowledgeAnswerTool:
    def __init__(self):
        self.responder = BedrockKnowledgeResponder()

    def run(self, prompt, history_text=""):
        return self.responder.answer(prompt, history_text=history_text)
