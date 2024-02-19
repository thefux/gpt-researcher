# from rich import print
# from rich.prompt import Prompt

# import langroid.language_models as lm
# from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig, ChatDocument
# from langroid.agent.task import Task
# from langroid.agent.tools.google_search_tool import GoogleSearchTool
# from langroid.agent.tools.duckduckgo_search_tool import DuckduckgoSearchTool
# from langroid.utils.configuration import Settings, set_global


# async def langroid_agent(
#     query: str,
#     debug: bool = False,
#     model = None,
#     provider: str = "ddg",
#     no_stream: bool = False,
#     nocache: bool = False,
# ) -> ChatDocument | None:
#     set_global(
#         Settings(
#             debug=debug,
#             cache=not nocache,
#             stream=not no_stream,
#         )
#     )
#     print(
#         """
#         [blue]Welcome to the Web Search chatbot!
#         I will try to answer your questions, relying on (summaries of links from)
#         Search when needed.

#         Enter x or q to quit at any point.
#         """
#     )
#     sys_msg = Prompt.ask(
#         "[blue]Tell me who I am. Hit Enter for default, or type your own\n",
#         default="Default: 'You are a helpful assistant'",
#     )

#     llm_config = lm.OpenAIGPTConfig(
#         chat_model=model or lm.OpenAIChatModel.GPT4_TURBO,
#         chat_context_length=8_000,
#         temperature=0,
#         max_output_tokens=200,
#         timeout=45,
#     )

#     config = ChatAgentConfig(
#         system_message=sys_msg,
#         llm=llm_config,
#         vecdb=None,
#     )
#     agent = ChatAgent(config)

#     match provider:
#         case "google":
#             search_tool_class = GoogleSearchTool
#         case "sciphi":
#             from langroid.agent.tools.sciphi_search_rag_tool import SciPhiSearchRAGTool

#             search_tool_class = SciPhiSearchRAGTool
#         case "metaphor":
#             from langroid.agent.tools.metaphor_search_tool import MetaphorSearchTool

#             search_tool_class = MetaphorSearchTool
#         case "ddg":
#             search_tool_class = DuckduckgoSearchTool
#         case _:
#             raise ValueError(f"Unsupported provider {provider} specified.")

#     agent.enable_message(search_tool_class)
#     search_tool_handler_method = search_tool_class.default_value("request")

#     task = Task(
#         agent,
#         system_message=f"""
#         You are a helpful assistant. You will try your best to answer my questions.
#         Here is how you should answer my questions:
#         - IF my question is about a topic you ARE CERTAIN about, answer it directly
#         - OTHERWISE, use the `{search_tool_handler_method}` tool/function-call to
#           get up to 5 results from a web-search, to help you answer the question.
#           I will show you the results from the web-search, and you can use those
#           to answer the question.
#         - If I EXPLICITLY ask you to search the web/internet, then use the
#             `{search_tool_handler_method}` tool/function-call to get up to 5 results
#             from a web-search, to help you answer the question.

#         Be very CONCISE in your responses, First show me your answer,
#         and then show me the SOURCE(s) and EXTRACT(s) to justify your answer,
#         in this format:

#         <your answer here>
#         SOURCE: https://www.wikihow.com/Be-a-Good-Assistant-Manager
#         EXTRACT: Be a Good Assistant ... requires good leadership skills.

#         SOURCE: ...
#         EXTRACT: ...

#         For the EXTRACT, ONLY show up to first 3 words, and last 3 words.
#         DO NOT MAKE UP YOUR OWN SOURCES; ONLY USE SOURCES YOU FIND FROM A WEB SEARCH.
#         """,
#     )

#     return await task.run_async(query)
