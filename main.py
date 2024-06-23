# Import necessary libraries and modules
from langchain import memory
import openai
import langchain
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain.memory import ConversationBufferMemory

# Load your API key from environment variables
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize the language model (ChatAnthropic in this case)
llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0.2,
    max_tokens=1024
)

def practice():
    """
    Example function demonstrating usage of LangChain for generating responses based on user input.
    """
    # List of user inputs (sample queries)
    user_inputs = [
        "how can langsmith help with testing?",
        "what are the benefits of using langsmith for developers?",
        "how does langsmith integrate with CI/CD pipelines?",
        "can langsmith be used for automated testing?",
        "what are the key features of langsmith?",
        "how can langsmith improve code quality?",
        "what languages does langsmith support?",
        "how do I get started with langsmith?",
        "are there any tutorials available for langsmith?",
        "how does langsmith handle large projects?"
    ]

    # Define a prompt template for interaction with the language model
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a world-class technical documentation writer."),
        ("user", "{input}")
    ])

    # Combine the prompt template with the language model
    chain = prompt | llm

    # Example usage: invoking the chain with a specific user input
    response = chain.invoke({"input": "how do I get started with langsmith?"})
    print(f"User Input: 'how do I get started with langsmith?'\nResponse: {response}\n")

def implementation():
    """
    Example function demonstrating usage of LangChain for creating a sequential chain
    to write a play synopsis and a review.
    """
    # Define a template for writing a synopsis given a title of a play
    synopsis_template = """You are a playwright. Given the title of the play, 
    it is your job to write a synopsis for it.\n\nTitle: {title}\nPlaywright: 
    This is a synopsis for the above play:"""
    
    # Define a prompt template for the synopsis
    synopsis_prompt_template = PromptTemplate(
        input_variables=["title"],
        template=synopsis_template
    )
    
    # Create an LLMChain for writing a synopsis
    synopsis_chain = LLMChain(llm=llm, prompt=synopsis_prompt_template)

    # Define a template for writing a review of a play given a synopsis
    review_template = """You are a play critic from the New York Times. Given the synopsis of the play, 
    it is your job to write a review for it.\n\nPlay Synopsis:\n{synopsis}\nReview from a New York Times 
    play critic of the above play:"""
    
    # Define a prompt template for the review
    review_prompt_template = PromptTemplate(
        input_variables=["synopsis"],
        template=review_template
    )

    # Create an LLMChain for writing a review
    review_chain = LLMChain(llm=llm, prompt=review_prompt_template)

    # Create a SimpleSequentialChain to link synopsis and review chains
    overall_chain = SimpleSequentialChain(
        chains=[synopsis_chain, review_chain],
        verbose=True  # Print verbose output for better understanding
    )

    # Example usage: running the overall chain with a specific play title
    review = overall_chain.run("Tragedy at sunset on the beach")
    print("Generated Play Review:", review)

def product_description_example():
    """
    Example function demonstrating usage of LangChain for generating a product description.
    """
    # Template for creating a product description
    product_description_template = """You are a marketing specialist. Given the details of the product, 
    write a compelling description for it.\n\nDescription: {details}"""
    
    # Define the prompt template for product description
    prompt_template_product_description = PromptTemplate(
        input_variables=["details"],
        template=product_description_template
    )

    # Create an LLMChain for generating a product description
    product_description_chain = LLMChain(
        llm=llm,
        prompt=prompt_template_product_description,
        memory=ConversationBufferMemory()  # Enable memory to track context
    )

    # Example product details (replace with actual product information)
    product_details = {
        "details": "Voice-controlled assistant that helps manage smart home devices."
    }

    # Invoke the chain with product details to generate a description
    response_product_description = product_description_chain.invoke(product_details)
    print(f"Product Description:\n{response_product_description}")

# Entry point for executing the functions
if __name__ == "__main__":
    # Uncomment the function you want to execute
    # practice()  # Example of user interaction
    # implementation()  # Example of writing play synopsis and review
    product_description_example()  # Example of generating a product description