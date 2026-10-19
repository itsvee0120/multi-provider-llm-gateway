import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from src.gateway import UnifiedLLMGateway

CASES = [
    ("ANTHROPIC_API_KEY", "claude-haiku-5-5"),
    ("GROQ_API_KEY", "openai/gpt-oss-20b"),
    # ("OPENAI_API_KEY", "<model>"),  # add later if you get a key
]


async def main():
    gw = UnifiedLLMGateway()
    for key_var, model in CASES:
        if not os.getenv(key_var):
            print(f"\n--- skipping {model}: {key_var} not set ---")
            continue
        print(f"\n=== {model} ===")
        try:
            async for chunk in gw.stream("Say hello in five words.", model, temperature=0.7):
                print(chunk.delta_text, end="", flush=True)
                if chunk.is_final:
                    print("\n", chunk.telemetry)
        except Exception as e:
            print(f"\nFAILED: {type(e).__name__}: {e}")


asyncio.run(main())