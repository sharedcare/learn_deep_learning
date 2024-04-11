import autogen


config_list = [
    {
        "model": "model_name",
        "api_base": "http://127.0.0.1:8001/v1"
        "api_type": "open_ai",
        "api_key": "NULL",
    }
]

agent_config={
    "request_timeout": 600,
    "seed": 42,
    "config_list": config_list,
    "temperature": 0
}

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=agent_config,
    system_message="Assistant agent"
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "agents-workspace"},
    llm_config=agent_config,
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet."""
)

user_proxy.initiate_chat(
    assistant,
    message="""Create a posting schedule with captions in instagram for a week and store it in a .csv file.""",
)