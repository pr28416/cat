#!/usr/bin/env python3

import os
import sys
import openai
import dotenv
import time
from typing import List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
dotenv.load_dotenv(os.path.join(PROJECT_ROOT, ".env.local"))


def get_bad_api_key() -> str:
    """Extract the specific bad API key."""
    # Check comma-separated format
    multi_keys_str = os.getenv("OPENAI_API_KEYS")
    if multi_keys_str:
        api_keys = [key.strip() for key in multi_keys_str.split(",")]
        # Find the key ending with Dj3DwXAp9L
        for key in api_keys:
            if key.endswith("Dj3DwXAp9L"):
                return key
    return None


def test_bad_key_multiple_times(api_key: str, num_tests: int = 10):
    """Test the bad API key multiple times with different models and requests."""
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")

    print(f"=== Testing Bad API Key Multiple Times ===")
    print(f"Key: {api_key[:20]}...{api_key[-10:]}")
    print(f"Base URL: {base_url}")
    print(f"Number of tests: {num_tests}")
    print()

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    # Test different models and prompts
    test_cases = [
        {"model": "gpt-4o-mini-2024-07-18", "prompt": "Say hello"},
        {"model": "gpt-4o", "prompt": "Count to 3"},
        {"model": "gpt-4.1-2025-04-14", "prompt": "What is 2+2?"},
        {"model": "gpt-4o-mini-2024-07-18", "prompt": "Name a color"},
        {"model": "gpt-4o", "prompt": "Say test"},
    ]

    results = []

    for i in range(num_tests):
        test_case = test_cases[i % len(test_cases)]
        model = test_case["model"]
        prompt = test_case["prompt"]

        print(f"Test #{i+1}: {model} - '{prompt}'")

        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10,
            )
            latency = time.time() - start_time

            content = response.choices[0].message.content
            result = {
                "test_num": i + 1,
                "model": model,
                "prompt": prompt,
                "status": "SUCCESS",
                "response": content,
                "latency": latency,
                "error": None,
            }
            print(f"  ✅ SUCCESS: '{content}' ({latency:.2f}s)")

        except Exception as e:
            latency = time.time() - start_time
            error_msg = str(e)
            result = {
                "test_num": i + 1,
                "model": model,
                "prompt": prompt,
                "status": "ERROR",
                "response": None,
                "latency": latency,
                "error": error_msg,
            }
            print(f"  ❌ ERROR: {error_msg}")

        results.append(result)

        # Small delay between requests
        time.sleep(0.5)

    # Analysis
    print("\n" + "=" * 50)
    print("ANALYSIS")
    print("=" * 50)

    successes = [r for r in results if r["status"] == "SUCCESS"]
    errors = [r for r in results if r["status"] == "ERROR"]

    print(f"Total tests: {len(results)}")
    print(f"Successes: {len(successes)}")
    print(f"Errors: {len(errors)}")
    print(f"Success rate: {len(successes)/len(results)*100:.1f}%")
    print(f"Error rate: {len(errors)/len(results)*100:.1f}%")

    if errors:
        print(f"\nError patterns:")
        error_types = {}
        for error in errors:
            error_type = (
                error["error"][:100] + "..."
                if len(error["error"]) > 100
                else error["error"]
            )
            error_types[error_type] = error_types.get(error_type, 0) + 1

        for error_type, count in error_types.items():
            print(f"  '{error_type}': {count} times")

    if successes:
        print(f"\nSuccessful models:")
        model_successes = {}
        for success in successes:
            model = success["model"]
            model_successes[model] = model_successes.get(model, 0) + 1

        for model, count in model_successes.items():
            print(f"  {model}: {count} times")

        avg_latency = sum(s["latency"] for s in successes) / len(successes)
        print(f"  Average latency: {avg_latency:.2f}s")

    return results


def main():
    # Get the bad API key
    bad_key = get_bad_api_key()

    if not bad_key:
        print("❌ Could not find the bad API key ending with 'Dj3DwXAp9L'")
        print("Available keys:")
        multi_keys_str = os.getenv("OPENAI_API_KEYS")
        if multi_keys_str:
            api_keys = [key.strip() for key in multi_keys_str.split(",")]
            for i, key in enumerate(api_keys):
                print(f"  Key #{i+1}: {key[:20]}...{key[-10:]}")
        return

    # Test the bad key multiple times
    test_bad_key_multiple_times(bad_key, num_tests=15)


if __name__ == "__main__":
    main()
