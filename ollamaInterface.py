from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

class CustomOllama:
    """Class to handle interactions with an Ollama LLM instance."""

    def __init__(self, model_name: str = "llama3.2:latest"):
        """
        Initialize the OllamaLLM instance.

        Args:
            model_name (str): The name of the model to use.
        """
        self.model = OllamaLLM(model=model_name)

    def send_prompt(self, template: str, question: str) -> str:
        """
        Send a prompt to the Ollama LLM for response.

        Args:
            template (str): The template to use for the prompt.
            question (str): The question or input to the LLM.

        Returns:
            str: The response from the LLM.
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.model
        return chain.invoke({"question": question})
