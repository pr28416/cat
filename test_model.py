#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()
print('Testing gpt-5-nano model...')
try:
    llm = ChatOpenAI(model='gpt-5-nano', temperature=0.3, timeout=15.0, max_retries=1)
    response = llm.invoke([HumanMessage(content='Say hello in 3 words')])
    print(f'✅ SUCCESS: {response.content}')
except Exception as e:
    print(f'❌ FAILED: {type(e).__name__}: {e}')
