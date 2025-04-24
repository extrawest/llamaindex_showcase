import os
import asyncio
import time
import logging
from typing import Optional

from llama_index.llms.mistralai import MistralAI
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.core.tools import QueryEngineTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.mistralai import MistralAIEmbedding

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mistral_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

def get_mistral_api_key() -> str:
    """Get Mistral API key from environment variable."""
    api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY environment variable is not set. "
            "Please set it in your .env file or pass it directly to RAGApp."
        )
    return api_key

llm = MistralAI(
    model="mistral-large-latest", 
    api_key=get_mistral_api_key(),
    max_retries=3,
    timeout=30
)

agent = FunctionAgent(
    tools=[multiply, add],
    llm=llm,
)

async def run_with_retry(ai_agent, query, ctx: Optional[Context]):
    """Run ai_agent with retry logic for rate limiting"""
    for attempt in range(3):
        try:
            logger.info(f"Attempting query: {query}")
            start_time = time.time()

            await asyncio.sleep(2)

            if ctx is None:
                response = await ai_agent.run(query)
            else:
                response = await ai_agent.run(query, ctx=ctx)

            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Query completed in {duration:.2f} seconds")
            
            return response
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < 3 - 1:
                wait_time = 5 * (2 ** attempt)
                logger.warning(f"Rate limit hit, waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Error in query: {str(e)}")
                raise e

async def main():
    try:
        logger.info("Starting Mistral Agent examples")

        logger.info("\nExample 1: Simple calculation")
        response = await run_with_retry(agent, "What is (121 + 2) * 5?", None)
        logger.info(f"Response: {str(response)}")

        await asyncio.sleep(2)

        logger.info("\nExample 2: Using context/memory")
        ctx = Context(agent)
        response = await run_with_retry(agent, "My name is John Doe", ctx)
        logger.info(f"Response: {str(response)}")
        response = await run_with_retry(agent, "What is my name?", ctx)
        logger.info(f"Response: {str(response)}")

        await asyncio.sleep(2)

        logger.info("\nExample 3: RAG Pipeline with Uber 10-K")
        embed_model = MistralAIEmbedding(
            api_key=get_mistral_api_key(),
            max_retries=3,
            timeout=30
        )
        query_llm = MistralAI(
            model="mistral-medium", 
            api_key=get_mistral_api_key(),
            max_retries=3,
            timeout=30
        )

        logger.info("Loading Uber 10-K document")
        uber_docs = SimpleDirectoryReader(
            input_files=["data/10k/uber_2021.pdf"]
        ).load_data()
        logger.info(f"Loaded {len(uber_docs)} documents")

        logger.info("Building vector index")
        uber_index = VectorStoreIndex.from_documents(
            uber_docs, embed_model=embed_model
        )
        uber_engine = uber_index.as_query_engine(similarity_top_k=3, llm=query_llm)

        query_engine_tool = QueryEngineTool.from_defaults(
            query_engine=uber_engine,
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        )

        logger.info("Creating Uber agent with RAG tool")
        uber_agent = FunctionAgent(tools=[query_engine_tool], llm=llm)

        logger.info("Querying about risk factors and tailwinds")
        response = await run_with_retry(
            uber_agent,
            "Tell me both the risk factors and tailwinds for Uber? Do two parallel tool calls.",
            None
        )
        logger.info(f"Response: {str(response)}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Starting Mistral Agent script")
    asyncio.run(main()) 