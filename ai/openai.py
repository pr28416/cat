import openai
import os
import dotenv
import random
import time
from typing import Dict, List, Any, Optional, Union, TypeVar, Type
from pydantic import BaseModel

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Load .env.local from the project root
dotenv.load_dotenv(os.path.join(PROJECT_ROOT, ".env.local"))


def get_openai_client():
    """Get an OpenAI client with API keys from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")  # Try single key format first

    if not api_key:
        # Try comma-separated format
        api_keys_str = os.getenv("OPENAI_API_KEYS")
        if api_keys_str:
            api_keys = [key.strip() for key in api_keys_str.split(",")]

            # Filter out known bad keys
            bad_key_suffixes = [
                "Dj3DwXAp9L",
                "4U1uixCjw7",
            ]  # Key #7 that fails, Key #5 rate limited
            good_keys = [
                key
                for key in api_keys
                if not any(key.endswith(suffix) for suffix in bad_key_suffixes)
            ]

            if good_keys:
                api_key = random.choice(good_keys)
            else:
                # Fallback to original list if filtering removes all keys
                api_key = random.choice(api_keys)

    if not api_key:
        raise ValueError(
            "No OpenAI API key found in environment variables. Please set OPENAI_API_KEY or OPENAI_API_KEYS."
        )

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")

    # Configure the client with default headers
    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    return client


def generate_text(
    prompt: str,
    model: str = "gpt-4.1",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    instructions: Optional[str] = None,
) -> str:
    """
    Generate text using the OpenAI Chat Completions API with retry logic.

    Args:
        prompt: The user input prompt
        model: The model to use for generation
        temperature: Controls randomness (0-1)
        max_tokens: Maximum tokens to generate
        instructions: Optional system instructions

    Returns:
        Generated text response
    """
    client = get_openai_client()
    max_retries = 5
    base_delay = 1  # seconds

    messages = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    messages.append({"role": "user", "content": prompt})

    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens:
        # o1/o3/o4 models use max_completion_tokens instead of max_tokens
        if model.startswith(("o1-", "o3-", "o4-")):
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**params)
            return response.choices[0].message.content
        except (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        ) as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                print(
                    f"API Error: {e}. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)
            else:
                print(f"API Error: {e}. All retries failed.")
                raise
    # This line should not be reachable, but as a fallback:
    raise Exception("All retries failed without catching a specific exception.")


def generate_text_with_messages(
    messages: List[Dict[str, str]],
    model: str = "gpt-4.1",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Generate text using the OpenAI Chat Completions API with message format and retry logic.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: The model to use for generation
        temperature: Controls randomness (0-1)
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text response
    """
    client = get_openai_client()
    max_retries = 5
    base_delay = 1  # seconds

    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens:
        # o1/o3/o4 models use max_completion_tokens instead of max_tokens
        if model.startswith(("o1-", "o3-", "o4-")):
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**params)
            return response.choices[0].message.content
        except (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        ) as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                print(
                    f"API Error: {e}. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)
            else:
                print(f"API Error: {e}. All retries failed.")
                raise
    # This line should not be reachable, but as a fallback:
    raise Exception("All retries failed without catching a specific exception.")


T = TypeVar("T", bound=BaseModel)


def generate_structured_output(
    prompt: Union[str, List[Dict[str, str]]],
    model_schema: Type[T],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    instructions: Optional[str] = None,
) -> T:
    """
    Generate structured output using the OpenAI Chat Completions API with Pydantic models.

    Args:
        prompt: Either a string prompt or list of message dictionaries
        model_schema: Pydantic BaseModel class defining the expected structure
        model: The model to use for generation (must support structured outputs)
        temperature: Controls randomness (0-1)
        max_tokens: Maximum tokens to generate
        instructions: Optional system instructions (if prompt is a string)

    Returns:
        Instance of the provided Pydantic model with parsed data
    """
    client = get_openai_client()

    messages = []
    if isinstance(prompt, str):
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": prompt})
    else:
        messages = prompt

    # Add schema information to the system message
    schema_message = {
        "role": "system",
        "content": f"Please provide your response in the following JSON schema: {model_schema.schema_json()}",
    }
    messages.insert(0, schema_message)

    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens:
        # o1/o3/o4 models use max_completion_tokens instead of max_tokens
        if model.startswith(("o1-", "o3-", "o4-")):
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens

    response = client.chat.completions.create(**params)
    content = response.choices[0].message.content

    # Parse the response into the Pydantic model
    try:
        return model_schema.parse_raw(content)
    except Exception as e:
        raise ValueError(f"Failed to parse response into {model_schema.__name__}: {e}")


if __name__ == "__main__":
    # Test the OpenAI client with a simple API call
    print("Testing OpenAI client...")

    # Debug environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
    print(f"API Key exists: {bool(api_key)}")
    print(f"API Key: {api_key}")
    print(f"API Key length: {len(api_key) if api_key else 0}")
    print(f"Base URL: {base_url}")

    try:
        client = get_openai_client()
        print("Successfully connected to OpenAI API!")

        # Test a simple text generation using responses API
        test_prompt = "Say hello in one word."
        print(f"\nTesting text generation with prompt: '{test_prompt}'")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.7,
            max_tokens=10,
        )
        print(f"Response: {response.choices[0].message.content}")

    except Exception as e:
        print(f"Error testing OpenAI client: {e}")
