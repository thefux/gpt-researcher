# from langchain_community.llms import Ollama

# llm = Ollama(model='llama2')

# from crewai import Agent

# def agent_crewai(topic: str):
#     # Creating a senior researcher agent
#     researcher = Agent(
#       role='Senior Researcher',
#       goal=f'Uncover groundbreaking technologies around {topic}',
#       verbose=True,
#       backstory="""Driven by curiosity, you're at the forefront of
#       innovation, eager to explore and share knowledge that could change
#       the world."""
#     )

#     # Creating a writer agent
#     writer = Agent(
#       role='Writer',
#       goal=f'Narrate compelling tech stories around {topic}',
#       verbose=True,
#       backstory="""With a flair for simplifying complex topics, you craft
#       engaging narratives that captivate and educate, bringing new
#       discoveries to light in an accessible manner."""
#     )

#     from crewai import Task

#     # Install duckduckgo-search for this example:
#     # !pip install -U duckduckgo-search

#     from langchain_community.tools import DuckDuckGoSearchRun
#     search_tool = DuckDuckGoSearchRun()

#     # Research task for identifying AI trends
#     research_task = Task(
#       description=f"""Identify the topic {topic}.
#       Your final report should clearly articulate the key points.
#       """,
#       tools=[search_tool],
#       agent=researcher
#     )

#     # Writing task based on research findings
#     write_task = Task(
#       description=f"""Compose an insightful samll article on {topic}.
#       """,
#       tools=[search_tool],
#       agent=writer
#     )

#     from crewai import Crew, Process

#     # Forming the tech-focused crew
#     crew = Crew(
#       agents=[researcher],
#       tasks=[research_task],
#       process=Process.sequential  # Sequential task execution
#     )

#     return crew.kickoff()
