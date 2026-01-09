from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langgraph.graph import StateGraph, MessagesState, START, END

def invoke_bedrock_model(params: dict = {}, prompt: str = ""):
    
    pass

def supervisor(state: MessagesState):
    system_prompt = SystemMessagePromptTemplate.from_template()
    user_prompt = HumanMessagePromptTemplate.from_template()
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

    formatted_prompt = chat_prompt.format_prompt(messages=state.messages)
    response = invoke_bedrock_model(prompt=formatted_prompt)

    return {"messages": state.messages + [{"role": "ai", "content": "Please be more polite."}]}

def color_receommeder_agent(state: MessagesState):
    user_message = state.messages[-1]["content"].lower()
    if "red" in user_message:
        color = "red"
    elif "blue" in user_message:
        color = "blue"
    else:
        color = "green"
    return {"messages": state.messages + [{"role": "ai", "content": f"I recommend the color {color}."}]}

def clothing_recommender_agent(state: MessagesState):
    user_message = state.messages[-1]["content"].lower()
    if "shirt" in user_message:
        clothing = "a stylish shirt"
    elif "pants" in user_message:
        clothing = "comfortable pants"
    else:
        clothing = "a nice jacket"
    return {"messages": state.messages + [{"role": "ai", "content": f"I suggest you wear {clothing}."}]}

def footwear_recommender_agent(state: MessagesState):
    user_message = state.messages[-1]["content"].lower()
    if "sneakers" in user_message:
        footwear = "sneakers"
    elif "boots" in user_message:
        footwear = "boots"
    else:
        footwear = "sandals"
    return {"messages": state.messages + [{"role": "ai", "content": f"You should consider wearing {footwear}."}]}

def final_recommender_agent(state: MessagesState):
    pass

def check_color_input(state: MessagesState) -> Literal["color_available", "color_unvailable"]:
    supervisor_state_message = json.loads(state["messages"][1].content)
    if supervisor_state_message["color"] == "not_available":
        return "color_available"
    
    else:
        return "color_unvailable"

if __name__ == "__main__":
    graph = StateGraph(MessagesState)
    graph.add_node(supervisor, name="supervisor_llm")
    graph.add_edge(START, "supervisor_llm")
    graph.add_edge("mock_llm", END)
    graph = graph.compile()

    print(graph.invoke({"messages": [{"role": "user", "content": "hi!"}]}))

