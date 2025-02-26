from ai_autonomy.ai_developer import AIDeveloper
import asyncio

async def main():
    ai_developer = AIDeveloper()
    await ai_developer.run()

if __name__ == "__main__":
    asyncio.run(main())