import datetime
from gpt_researcher.master.functions import stream_output
from gpt_researcher.master.agent import GPTResearcher

async def run_agent(task, report_type):
    """Run the agent."""
    # measure time
    start_time = datetime.datetime.now()
    # add customized JSON config file path here
    config_path = None
    # run agent
    researcher = GPTResearcher(query=task, report_type=report_type, source_urls=None, config_path=config_path)
    report = await researcher.run()
    # measure time
    end_time = datetime.datetime.now()
    await stream_output("logs", f"\nTotal run time: {end_time - start_time}\n")

    return report
