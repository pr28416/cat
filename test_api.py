#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
print(f'API Key present: {bool(api_key)}')
if api_key:
    print(f'API Key starts with: {api_key[:10]}...')

# Try a simple API call
try:
    llm = ChatOpenAI(model='gpt-5-nano', temperature=0.3, timeout=10.0)
    response = llm.invoke([HumanMessage(content='Hello')])
    print(f'✅ API call successful: {response.content[:50]}...')
except Exception as e:
    print(f'❌ API call failed: {type(e).__name__}: {e}')
