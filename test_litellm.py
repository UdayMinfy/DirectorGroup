#!/usr/bin/env python3
import sys
sys.path.insert(0, '/root/work/lambdacode2/src')

try:
    print("=== Testing LiteLLM Installation ===")
    import litellm
    print(f"✓ litellm imported successfully")
    
    print("\n=== Testing token_counter function ===")
    from litellm import token_counter
    print(f"✓ token_counter function imported")
    
    # Test with messages format (for input tokens)
    print("\n--- Testing with messages format ---")
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    try:
        tokens = token_counter(model="claude-3-sonnet", messages=messages)
        print(f"✓ token_counter(messages, claude-3-sonnet): {tokens} tokens")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test with text format (for output tokens)
    print("\n--- Testing with text format ---")
    test_text = "The capital of France is Paris. It is known for the Eiffel Tower and many other historical landmarks."
    try:
        tokens = token_counter(model="claude-3-sonnet", text=test_text)
        print(f"✓ token_counter(text, claude-3-sonnet): {tokens} tokens")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test system prompt
    print("\n--- Testing system prompt ---")
    system = "You are a helpful assistant."
    try:
        tokens = token_counter(model="claude-3-sonnet", text=system)
        print(f"✓ token_counter(system_text): {tokens} tokens")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test the actual agent code
    print("\n=== Testing Agent Integration ===")
    from simplified_agent import SimplifiedResearchAgent
    agent = SimplifiedResearchAgent()
    
    # Simulate building usage
    system_prompt = "You are a helpful assistant."
    user_prompt = "What is Python?"
    output_text = "Python is a high-level programming language."
    
    usage = agent._build_usage(system_prompt, user_prompt, output_text)
    print(f"✓ Agent._build_usage() returned: {usage}")
    
    if usage["input_tokens"] > 0 and usage["output_tokens"] > 0:
        print("✓ Token counting is working correctly!")
    else:
        print(f"! WARNING: Got zero tokens - input={usage['input_tokens']}, output={usage['output_tokens']}")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

