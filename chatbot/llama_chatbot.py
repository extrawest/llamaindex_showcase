import os
from pathlib import Path
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("../data")
STORAGE_DIR = Path("../storage")

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import UnstructuredReader

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

Settings.llm = OpenAI(model="gpt-4")
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-large")
Settings.chunk_size = 512
Settings.chunk_overlap = 64

def setup_directories():
    """Create necessary directories if they don't exist."""
    DATA_DIR.mkdir(exist_ok=True)
    STORAGE_DIR.mkdir(exist_ok=True)

def load_or_create_indices(years: list[int]) -> dict:
    """Load existing indices or create new ones for each year."""
    index_set = {}
    
    for year in years:
        year_storage_dir = STORAGE_DIR / str(year)
        html_file = DATA_DIR / f"UBER_{year}.html"
        
        if not html_file.exists():
            logger.error(f"HTML file not found: {html_file}")
            continue
            
        if year_storage_dir.exists():
            logger.info(f"Loading index for year {year}")
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(year_storage_dir)
                )
                cur_index = load_index_from_storage(storage_context)
                docstore = cur_index.docstore
                logger.info(f"Index for year {year} contains {len(docstore.docs)} documents")
                logger.info(f"First few document IDs: {list(docstore.docs.keys())[:5]}")
                logger.info(f"Successfully loaded index for year {year}")
            except Exception as e:
                logger.error(f"Error loading index for year {year}: {str(e)}")
                continue
        else:
            logger.info(f"Creating index for year {year}")
            try:
                loader = UnstructuredReader()
                logger.info(f"Loading HTML file: {html_file}")
                year_docs = loader.load_data(
                    file=html_file,
                    split_documents=False
                )
                logger.info(f"Successfully loaded {len(year_docs)} documents")

                for i, doc in enumerate(year_docs[:3]):
                    logger.info(f"Document {i} preview: {doc.text[:200]}...")

                for doc in year_docs:
                    doc.metadata = {"year": year}

                storage_context = StorageContext.from_defaults()
                cur_index = VectorStoreIndex.from_documents(
                    year_docs,
                    storage_context=storage_context,
                )
                storage_context.persist(persist_dir=str(year_storage_dir))
                logger.info(f"Successfully created and persisted index for year {year}")
            except Exception as e:
                logger.error(f"Error creating index for year {year}: {str(e)}")
                continue
        
        index_set[year] = cur_index
    
    if not index_set:
        raise ValueError("No indices were successfully loaded or created")
    
    return index_set

def setup_query_engines(index_set: dict) -> SubQuestionQueryEngine:
    """Set up query engines for individual years and combined analysis."""
    individual_query_engine_tools = []
    
    for year in index_set.keys():
        logger.info(f"Setting up query engine for year {year}")
        try:
            query_engine = index_set[year].as_query_engine(
                similarity_top_k=5,
                response_mode="tree_summarize",
                verbose=True,
                streaming=True,
                timeout=30
            )

            tool = QueryEngineTool.from_defaults(
                query_engine=query_engine,
                name=f"vector_index_{year}",
                description=(
                    "useful for when you want to answer queries about the"
                    f" {year} SEC 10-K for Uber. Focus on risk factors,"
                    " financial performance, and business operations."
                ),
            )
            individual_query_engine_tools.append(tool)
            logger.info(f"Successfully created query engine tool for year {year}")
        except Exception as e:
            logger.error(f"Error creating query engine for year {year}: {str(e)}")
            continue
    
    if not individual_query_engine_tools:
        raise ValueError("No query engine tools were successfully created")

    logger.info("Setting up sub-question query engine")
    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=individual_query_engine_tools,
        verbose=True,
        use_async=True
    )
    logger.info("Successfully created sub-question query engine")
    
    return query_engine

def setup_agent(query_engine: SubQuestionQueryEngine) -> ReActAgent:
    """Set up the chatbot agent with all necessary tools."""
    query_engine_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="sub_question_query_engine",
        description=(
            "useful for when you want to answer queries that require analyzing"
            " multiple SEC 10-K documents for Uber. Focus on risk factors,"
            " financial performance, and business operations."
        ),
    )

    tools = [query_engine_tool]

    return ReActAgent.from_tools(
        tools=tools,
        llm=OpenAI(model="gpt-4", temperature=0.1),
        verbose=True,
        max_iterations=3
    )

async def chat_loop(agent: ReActAgent):
    """Run the interactive chat loop."""
    print("\nWelcome to the Uber 10-K Chatbot!")
    print("Type 'exit' to end the conversation.\n")
    
    while True:
        try:
            text_input = input("User: ")
            if text_input.lower() == "exit":
                break
                
            response = agent.chat(text_input)
            print(f"\nAgent: {response.response}\n")
            
        except Exception as e:
            logger.error(f"Error during chat: {str(e)}")
            print("Sorry, there was an error processing your request. Please try again.")

async def main():
    """Main function to set up and run the chatbot."""
    try:
        setup_directories()

        years = [2022, 2021, 2020, 2019]

        logger.info("Loading or creating indices...")
        index_set = load_or_create_indices(years)

        logger.info("Setting up query engines...")
        query_engine = setup_query_engines(index_set)

        logger.info("Setting up chatbot agent...")
        agent = setup_agent(query_engine)

        logger.info("Starting chat loop...")
        await chat_loop(agent)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 