import json
from typing import Literal
import boto3
from botocore.config import Config

from langchain_aws import ChatBedrock
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END

class BedrockClient:
    def __init__(self, region_name: str = "us-east-1"):
        self.client = boto3.client(service_name='bedrock-runtime', 
                                   region_name=region_name, 
                                   config=Config(retries={"max_attempts": 10}))

    def invoke_model(self, prompt: str, params: dict = {}):
        lc_client = ChatBedrock(client=self.client, model_id=params.get("model_id", "amazon.nova-micro-v1:0"))
        response = lc_client.invoke(prompt)

        return response
    
bedrock_client = BedrockClient()

def supervisor_agent(state: MessagesState):
    system_prompt = SystemMessagePromptTemplate.from_template(
        """# Instruction
        You are tasked with extracting value for key features from the input text. Key features include: gender, color, occasion, season or personal 
        fashion taste. If no value is found for respective categories, use "not_available"

        #Output 
        Use the following JSON format for output in all lowercase. Do not add any additional explanations, punctuations or reasonings. 
        ## JSON Format
        {{"gender": "gender mentioned in the text, else assume male", 
        "color": "color mentioned in the text for clothes or footwear",
        "occasion": "occasion that the clothes or footwear is intended for",
        "season": "season that is mentioned or occasion that usually occurs in particular season", 
        "personal_fashion_taste": "any additional fashion taste or clues in the inputs"}}
        """
    )

    user_prompt_template = HumanMessagePromptTemplate.from_template(
    """
    #Inputs
    input_text : {input_text}
    """
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt_template])

    formatted_prompt = chat_prompt.format_prompt(input_text=state["messages"][0].content)
    response = bedrock_client.invoke_model(prompt=formatted_prompt)

    return {"messages": [response]}

def color_recommender_agent(state: MessagesState):
    supervisor_state_message = json.loads(state["messages"][-1].content)

    system_prompt = SystemMessagePromptTemplate.from_template(
        """System Prompt
        # Instruction
        You are tasked with recommending the best color for clothes and footwear  based on the gender, season and occasion. 

        #Output 
        Use the following JSON format for output in all lowercase. Do not add any additional explanations, punctuations or reasonings. 
        ## JSON Format
        {{"color": ""}}
        """
    )

    user_prompt_template = HumanMessagePromptTemplate.from_template(
    """
    #Inputs
    gender: {gender}
    occasion: {occasion}
    season: {season}
    """
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt_template])

    formatted_prompt = chat_prompt.format_prompt(gender=supervisor_state_message.get("gender"), 
                                           occasion=supervisor_state_message.get("occasion"), 
                                           season=supervisor_state_message.get("season"))
    response = bedrock_client.invoke_model(formatted_prompt)

    return {"messages": [response]}

def clothing_recommender_agent(state: MessagesState):
    state_message = json.loads(state["messages"][-1].content)

    if state_message["color"] == "not_available":
        supervisor_state_message = json.loads(state["messages"][1].content)
        recommended_color = json.loads(state["messages"][2].content)["color"]
    
    else:
        supervisor_state_message = state_message
        recommended_color = supervisor_state_message["color"]

    system_prompt = SystemMessagePromptTemplate.from_template(
        """System Prompt
        # Instruction
        You are tasked with recommending the clothing and styling summary based on the available key features: gender, color, occasion, season or personal fashion taste.

        #Output 
        Use the following JSON format for output in all lowercase. Do not add any additional explanations, punctuations or reasonings. 
        ## JSON Format
        {{"clothing": "recommend and explain the styling in less than 30 words"}}
        """
    )

    user_prompt_template = HumanMessagePromptTemplate.from_template(
    """
    #Inputs
    gender: {gender}
    occasion: {occasion}
    season: {season}
    color: {color}
    personal_fashion_taste: {personal_fashion_taste}
    """
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt_template])

    formatted_prompt = chat_prompt.format_prompt(gender=supervisor_state_message.get("gender"), 
                                           occasion=supervisor_state_message.get("occasion"), 
                                           season=supervisor_state_message.get("season"),  
                                           color=recommended_color,  
                                           personal_fashion_taste=supervisor_state_message.get("personal_fashion_taste"))
    response = bedrock_client.invoke_model(formatted_prompt)

    return {"messages": [response]}

def router_agent(state: MessagesState) -> Literal["color_available", "color_unavailable"]:
    supervisor_state_message = json.loads(state["messages"][1].content)
    if supervisor_state_message["color"] == "not_available":
        return "color_unavailable"
    
    else:
        return "color_available"

if __name__ == "__main__":
    
    graph = StateGraph(MessagesState)

    # LangGraph Node Definitions
    graph.add_node(supervisor_agent, name="supervisor_agent")
    graph.add_node(router_agent, name="router_agent")
    graph.add_node(color_recommender_agent, name="color_recommender_agent")
    graph.add_node(clothing_recommender_agent, name="clothing_recommender_agent")

    # LangGraph Edge Definitions
    graph.add_edge(START, "supervisor_agent")

    graph.add_conditional_edges(
        "supervisor_agent", 
        router_agent, 
        {"color_unavailable": "color_recommender_agent", 
         "color_available": "clothing_recommender_agent"})
    
    graph.add_edge("color_recommender_agent", "clothing_recommender_agent")
    graph.add_edge("clothing_recommender_agent", END)

    # LangGraph Compilation
    graph = graph.compile()

    # LangGraph Input State and Graph Invoke
    input_text = "I am male looking for fashion ideas to attend my friend's wedding in Fall. I love Yellow color."
    input_state = {"messages": [HumanMessage(content=input_text)]}

    result = graph.invoke(input_state)
    
    print(f"Input user text: {input_text}")
    print(f"Agentic Response: {result["messages"][-1].content}")

    png_data = graph.get_graph().draw_mermaid_png()

    # Save the bytes to a file
    with open("outputs/graph_visualization.png", "wb") as f:
        f.write(png_data)