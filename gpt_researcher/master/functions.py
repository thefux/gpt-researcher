import asyncio

from litellm import ollama

from gpt_researcher.config.config import Config
from gpt_researcher.utils.llm import *
from gpt_researcher.scraper import Scraper
from gpt_researcher.master.prompts import *
import json

from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, set_global_service_context
from llama_index.core import ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

system_prompt = f"""
You are a helpful assistant. You will try your best to answer my questions.
Here is how you should answer my questions:
- use the contex provided to answer the question

Be very CONCISE in your responses, First show me your answer,
and then show me the SOURCE(s) and EXTRACT(s) to justify your answer,
in this format:

<your answer here>
SOURCE: https://www.wikihow.com/Be-a-Good-Assistant-Manager
EXTRACT: Be a Good Assistant ... requires good leadership skills.

SOURCE: ...
EXTRACT: ...

For the EXTRACT, ONLY show up to first 3 words, and last 3 words.
DO NOT MAKE UP YOUR OWN SOURCES; ONLY USE SOURCES YOU FIND FROM A WEB SEARCH.

YOU MUST WRITE THE REPORT WITH MARKDOWN SYNTAX.
"""
ollama_llm = Ollama(model='mistral', system_prompt=system_prompt)
ollama_embedding = OllamaEmbedding(model_name='nomic-embed-text')
openai_llm = OpenAI(system_prompt=system_prompt)

from llama_index.core import PromptHelper, ServiceContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser

def create_service_context(
    model,
    max_input_size = 4096,        # Context window for the LLM.
    num_outputs = 256,            # Number of output tokens for the LLM.
    chunk_overlap_ratio = 0.2,    # Chunk overlap as a ratio of chunk size.
    chunk_size_limit = None,      # Maximum chunk size to use.
    chunk_overlap = 30,           # Chunk overlap to use.
    chunk_size = 2048,            # Set chunk overlap to use.
    ):

    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    prompt_helper = PromptHelper(
        max_input_size,
        num_outputs,
        chunk_overlap_ratio,
        chunk_size_limit=chunk_size_limit)

    openai_embedding = OpenAIEmbedding()
    ollama_embedding = OllamaEmbedding(model_name='nomic-embed-text')

    service_context = ServiceContext.from_defaults(
        llm=model,
        embed_model=ollama_embedding,
        node_parser=node_parser,
        prompt_helper=prompt_helper)

    return service_context

set_global_service_context(create_service_context(openai_llm))


def get_retriever(retriever):
    """
    Gets the retriever
    Args:
        retriever: retriever name

    Returns:
        retriever: Retriever class

    """
    match retriever:
        case "tavily":
            from gpt_researcher.retrievers import TavilySearch
            retriever = TavilySearch
        case "tavily_news":
            from gpt_researcher.retrievers import TavilyNews
            retriever = TavilyNews
        case "google":
            from gpt_researcher.retrievers import GoogleSearch
            retriever = GoogleSearch
        case "searx":
            from gpt_researcher.retrievers import SearxSearch
            retriever = SearxSearch
        case "serpapi":
            raise NotImplementedError("SerpApiSearch is not fully implemented yet.")
            from gpt_researcher.retrievers import SerpApiSearch
            retriever = SerpApiSearch
        case "googleSerp":
            from gpt_researcher.retrievers import SerperSearch
            retriever = SerperSearch
        case "duckduckgo":
            from gpt_researcher.retrievers import Duckduckgo
            retriever = Duckduckgo
        case "BingSearch":
            from gpt_researcher.retrievers import BingSearch
            retriever = BingSearch

        case _:
            raise Exception("Retriever not found.")

    return retriever


async def choose_agent(query, cfg: Config):
    """
    Chooses the agent automatically
    Args:
        query: original query
        cfg: Config

    Returns:
        agent: Agent name
        agent_role_prompt: Agent role prompt
    """
    try:
        response = await create_chat_completion(
            model=cfg.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{auto_agent_instructions()}"},
                {"role": "user", "content": f"task: {query}"}],
            temperature=0,
            base_url=cfg.base_url
        )
        agent_dict = json.loads(response)
        return agent_dict["server"], agent_dict["agent_role_prompt"]
    except Exception as e:
        return "Default Agent", "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."


async def get_sub_queries(query, agent_role_prompt, cfg: Config):
    """
    Gets the sub queries
    Args:
        query: original query
        agent_role_prompt: agent role prompt
        cfg: Config

    Returns:
        sub_queries: List of sub queries

    """
    max_research_iterations = cfg.max_iterations if cfg.max_iterations else 1
    response = await create_chat_completion(
        model=cfg.smart_llm_model,
        messages=[
            {"role": "system", "content": f"{agent_role_prompt}"},
            {"role": "user", "content": generate_search_queries_prompt(query, max_iterations=max_research_iterations)}],
        temperature=0,
        base_url=cfg.base_url
    )
    sub_queries = json.loads(response)
    return sub_queries


def scrape_urls(urls, cfg=None):
    """
    Scrapes the urls
    Args:
        urls: List of urls
        cfg: Config (optional)

    Returns:
        text: str

    """
    content = []
    user_agent = cfg.user_agent if cfg else "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
    try:
        content = Scraper(urls, user_agent).run()
    except Exception as e:
        print(f"{Fore.RED}Error in scrape_urls: {e}{Style.RESET_ALL}")
    return content


async def summarize(query, content, agent_role_prompt, cfg, websocket=None):
    """
    Asynchronously summarizes a list of URLs.

    Args:
        query (str): The search query.
        content (list): List of dictionaries with 'url' and 'raw_content'.
        agent_role_prompt (str): The role prompt for the agent.
        cfg (object): Configuration object.

    Returns:
        list: A list of dictionaries with 'url' and 'summary'.
    """

    # Function to handle each summarization task for a chunk
    async def handle_task(url, chunk):
        summary = await summarize_url(query, chunk, agent_role_prompt, cfg)
        if summary:
            await stream_output("logs", f"🌐 Summarizing url: {url}", websocket)
            await stream_output("logs", f"📃 {summary}", websocket)
        return url, summary

    # Function to split raw content into chunks of 10,000 words
    def chunk_content(raw_content, chunk_size=10000):
        words = raw_content.split()
        for i in range(0, len(words), chunk_size):
            yield ' '.join(words[i:i+chunk_size])

    # Process each item one by one, but process chunks in parallel
    concatenated_summaries = []
    for item in content:
        url = item['url']
        raw_content = item['raw_content']

        # Create tasks for all chunks of the current URL
        chunk_tasks = [handle_task(url, chunk) for chunk in chunk_content(raw_content)]

        # Run chunk tasks concurrently
        chunk_summaries = await asyncio.gather(*chunk_tasks)

        # Aggregate and concatenate summaries for the current URL
        summaries = [summary for _, summary in chunk_summaries if summary]
        concatenated_summary = ' '.join(summaries)
        concatenated_summaries.append({'url': url, 'summary': concatenated_summary})

    return concatenated_summaries


async def summarize_url(query, raw_data, agent_role_prompt, cfg: Config):
    """
    Summarizes the text
    Args:
        query:
        raw_data:
        agent_role_prompt:
        cfg:

    Returns:
        summary: str

    """
    summary = ""
    try:
        summary = await create_chat_completion(
            model=cfg.fast_llm_model,
            messages=[
                {"role": "system", "content": f"{agent_role_prompt}"},
                {"role": "user", "content": f"{generate_summary_prompt(query, raw_data)}"}],
            temperature=0,
            base_url=cfg.base_url
        )
    except Exception as e:
        print(f"{Fore.RED}Error in summarize: {e}{Style.RESET_ALL}")
    return summary



async def generate_report(query, context, agent_role_prompt, report_type, websocket, cfg: Config, urls):
    """
    generates the final report
    Args:
        query:
        context:
        agent_role_prompt:
        report_type:
        websocket:
        cfg:

    Returns:
        report:

    """
    generate_prompt = get_report_by_type(report_type)
    report = "error while working on ur request"
    try:
        import validators
        from llama_index.readers.web import TrafilaturaWebReader
        from llama_index.readers.web import SimpleWebPageReader
        documents = []
        for url in list(urls):
            if validators.url(url):
                try:
                    # document = SimpleWebPageReader(html_to_text=True).load_data([url])
                    document = TrafilaturaWebReader().load_data([url], include_links=True)

                    await stream_output("logs", f"✅ done proccessing: {url}")

                    documents.extend(document)
                except Exception as e:
                    await stream_output("logs", f"Error: {e}")
                    continue


        # from llama_index.core.node_parser import SemanticSplitterNodeParser
        # from llama_index.embeddings.ollama import OllamaEmbedding
        # embeddings = OllamaEmbedding(model_name='nomic-embed-text')
        # splitter = SemanticSplitterNodeParser(
        #     buffer_size=1, embed_model=embeddings
        # )

        index = VectorStoreIndex.from_documents(documents)
        await stream_output("logs", f"✅ done indexing", websocket=websocket)

        query_engine = index.as_query_engine()
        prompt = f"{generate_prompt(query, context, 'markdown', cfg.total_words)}"
        response_result = await query_engine.aquery(prompt)

        report = response_result.response.rstrip()

        # report = await create_chat_completion(
        #     model=cfg.smart_llm_model,
        #     messages=[
        #         {"role": "system", "content": f"{system_prompt}"},
        #         {"role": "user", "content": f"{generate_prompt(query, context, cfg.report_format, cfg.total_words)}"}],
        #     temperature=0.2,
        #     stream=True,
        #     websocket=websocket,
        #     max_tokens=cfg.smart_token_limit,
        #     base_url=cfg.base_url
        # )

        print('report: ', report, type(report))
    except Exception as e:
        await stream_output("logs", f"{Fore.RED}Error in generate_report: {e}{Style.RESET_ALL}")

    return report


async def stream_output(type, output, websocket=None, logging=True):
    """
    Streams output to the websocket
    Args:
        type:
        output:

    Returns:
        None
    """
    if not websocket or logging:
        print(output)

    if websocket:
        await websocket.send_json({"type": type, "output": output})
