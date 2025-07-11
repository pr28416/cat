#!/usr/bin/env python3

import os
import sys
import openai
import dotenv
from typing import List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
dotenv.load_dotenv(os.path.join(PROJECT_ROOT, ".env.local"))


def get_all_api_keys() -> List[str]:
    """Extract all API keys from environment variables."""
    keys = []

    # Check single key format
    single_key = os.getenv("OPENAI_API_KEY")
    if single_key:
        keys.append(single_key)

    # Check comma-separated format
    multi_keys_str = os.getenv("OPENAI_API_KEYS")
    if multi_keys_str:
        multi_keys = [key.strip() for key in multi_keys_str.split(",")]
        keys.extend(multi_keys)

    # Remove duplicates while preserving order
    seen = set()
    unique_keys = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique_keys.append(key)

    return unique_keys


def test_api_key(api_key: str, key_index: int) -> dict:
    """Test a single API key with multiple models."""
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")

    print(f"\n--- Testing API Key #{key_index + 1} ---")
    print(f"Key: {api_key[:20]}...{api_key[-10:] if len(api_key) > 30 else api_key}")
    print(f"Base URL: {base_url}")

    # client = openai.OpenAI(api_key=api_key, base_url=base_url)
    client = openai.OpenAI(api_key=api_key)

    # Test models that we've been using
    test_models = [
        "gpt-4o-mini-2024-07-18",
        "gpt-4o",
        "gpt-4.1-2025-04-14",  # This one had issues before
    ]

    results = {
        "key_index": key_index + 1,
        "key_preview": f"{api_key[:20]}...{api_key[-10:] if len(api_key) > 30 else api_key}",
        "models": {},
    }

    for model in test_models:
        print(f"  Testing model: {model}")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say 'test' in one word."}],
                temperature=0.1,
                max_tokens=5,
            )

            content = response.choices[0].message.content
            results["models"][model] = {
                "status": "SUCCESS",
                "response": content,
                "error": None,
            }
            print(f"    ✅ SUCCESS: '{content}'")

        except Exception as e:
            error_msg = str(e)
            results["models"][model] = {
                "status": "ERROR",
                "response": None,
                "error": error_msg,
            }
            print(f"    ❌ ERROR: {error_msg}")

    return results


def main():
    print("=== API Key Testing Tool ===")

    # Get all API keys
    api_keys = get_all_api_keys()

    if not api_keys:
        print("❌ No API keys found in environment variables!")
        print("Please check OPENAI_API_KEY or OPENAI_API_KEYS in your .env.local file")
        return

    print(f"Found {len(api_keys)} API key(s) to test")

    # Test each key
    all_results = []
    for i, key in enumerate(api_keys):
        try:
            result = test_api_key(key, i)
            all_results.append(result)
        except Exception as e:
            print(f"❌ Failed to test key #{i + 1}: {e}")
            all_results.append(
                {
                    "key_index": i + 1,
                    "key_preview": f"{key[:20]}...{key[-10:] if len(key) > 30 else key}",
                    "models": {},
                    "connection_error": str(e),
                }
            )

    # Summary report
    print("\n" + "=" * 50)
    print("SUMMARY REPORT")
    print("=" * 50)

    for result in all_results:
        print(f"\nAPI Key #{result['key_index']} ({result['key_preview']}):")

        if "connection_error" in result:
            print(f"  ❌ CONNECTION ERROR: {result['connection_error']}")
            continue

        total_models = len(result["models"])
        successful_models = sum(
            1
            for model_result in result["models"].values()
            if model_result["status"] == "SUCCESS"
        )

        print(f"  Models tested: {total_models}")
        print(f"  Successful: {successful_models}/{total_models}")

        if successful_models == total_models:
            print(f"  ✅ FULLY FUNCTIONAL")
        elif successful_models > 0:
            print(f"  ⚠️  PARTIALLY FUNCTIONAL")
        else:
            print(f"  ❌ NOT FUNCTIONAL")

        # Show specific model failures
        for model, model_result in result["models"].items():
            if model_result["status"] == "ERROR":
                print(f"    ❌ {model}: {model_result['error']}")

    # Recommendations
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS")
    print("=" * 50)

    fully_functional_keys = []
    for result in all_results:
        if "connection_error" not in result:
            successful_models = sum(
                1
                for model_result in result["models"].values()
                if model_result["status"] == "SUCCESS"
            )
            total_models = len(result["models"])
            if successful_models == total_models:
                fully_functional_keys.append(result["key_index"])

    if fully_functional_keys:
        print(f"✅ Fully functional keys: {len(fully_functional_keys)}")
        print(f"   Key numbers: {fully_functional_keys}")
        print("   Recommendation: Use only these keys for experiments")
    else:
        print("❌ No fully functional keys found")
        print("   Recommendation: Check API key validity and quotas")


if __name__ == "__main__":
    main()
