from pydantic import BaseModel, Field, SecretStr
from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
from backend.assistants.assistant_types import AssistantState, Message
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
# from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama
# from ollama import AsyncClient

foundational_llms = {
    "gemini": ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-pro-latest", api_key=SecretStr(os.environ["GOOGLE_API_KEY"]))
    # open_ai_4o = ChatOpenAI(temperature=0, model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])
    # ollama_llama_31_8: ChatOllama(
#     model="llama3.1",
#     temperature=0,
# )
}

class AssistantConfig(BaseModel):
    llm_model: str = "gemini"
    llm_temperature: float = 0.0

class BaseAssistant(ABC, BaseModel):
    name: str # ?
    config: AssistantConfig = AssistantConfig()
    tools: List[Any] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    async def call(self, conversation_id: str, user_input: str) -> str:
        """Main entry point: process input, retrieve response, and log interaction."""
        state = AssistantState.query_by_id(conversation_id)
        if state is None:
            state = AssistantState(id=conversation_id)
        correspondent_message = Message(
            sent_by="correspondent",
            text=user_input,
            llm_name="",
            llm_model="",
            llm_temperature=0.0
        )
        state.messages.append(correspondent_message)
        response = await self.generate_response(state)
        
        assistant_message = Message(
            sent_by="assistant",
            text=response,
            llm_name=self.config.llm_model,
            llm_model=self.config.llm_model,
            llm_temperature=self.config.llm_temperature,
        )
        state.messages.append(assistant_message)

        state.save_to_db()
        return state.messages[-1].text

    @abstractmethod
    async def generate_response(self, state: AssistantState) -> str:
        """Generate response by querying the model or processing input. Defined in subclass."""
        raise NotImplementedError
    
    @abstractmethod
    async def handle_tool_call(self, response, function_call, tool_calls) -> str:
        raise NotImplementedError("base_assistant shouldnt trigger tools")
    
    async def llm_query(self, instruction: str, messages: list[Message], llm_name) -> BaseMessage:
        llm = foundational_llms[llm_name]
        if self.tools:
            llm = llm.bind_tools(self.tools, tool_choice="auto")
        system_inst = (
            "system",
            "Inst: {instruction}."
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                system_inst,
                ("human", "{chat_history}"),
            ]
        )
        chain = prompt | llm
        chat_history = self.get_chat_history(messages, 1) # Switched to 1 becasue charts are very long. Might need to filer ?! or summarize ! charts.
        prompt = {
            "instruction": instruction,
            "chat_history": chat_history,
        }
        response = chain.invoke(prompt)
        return response
    
    def get_chat_history(self, messages: list[Message], n: int) -> list[BaseMessage]:
        last_message = messages[-1]
        if not last_message.sent_by == "correspondent":
            raise ValueError("Last message must be from the correspondent")
        
        formated_messages = []
        for message in messages[-n:]:
            if message.sent_by == "assistant":
                formated_messages.append(AIMessage(content=message.text))
            elif message.sent_by == "correspondent":
                formated_messages.append(HumanMessage(content=message.text))

        return formated_messages
