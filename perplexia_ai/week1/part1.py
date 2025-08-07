"""Part 1 - Query Understanding implementation.

This implementation focuses on:
- Classify different types of questions
- Format responses based on query type
- Present information professionally
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from perplexia_ai.core.chat_interface import ChatInterface


class QueryType(Enum):
    """Enumeration of different query types for classification."""

    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARISON = "comparison"
    DEFINITION = "definition"


class QueryUnderstandingChat(ChatInterface):
    """Week 1 Part 1 implementation focusing on query understanding."""

    def __init__(self):
        self.llm: Optional[ChatOpenAI] = None
        self.query_classifier_prompt: Optional[ChatPromptTemplate] = None
        self.response_prompts: Dict[QueryType, ChatPromptTemplate] = {}
        self.classification_chain = None
        self.response_chains: Dict[QueryType, Any] = {}

    def initialize(self) -> None:
        """Initialize components for query understanding.

        Sets up:
        - Chat model
        - Query classification prompts
        - Response formatting prompts
        - Processing chains
        """
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

        self._setup_classifier_prompt()

        self._setup_response_prompts()

        self._setup_chains()

    def _setup_classifier_prompt(self) -> None:
        """Set up the query classification prompt template."""
        self.query_classifier_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a query classifier. Analyze the user's question and classify it into one of these categories:

1. FACTUAL: Direct questions asking for specific facts, data, or information
   - Examples: "What is the capital of France?", "Who invented the telephone?", "When did World War II end?"

2. ANALYTICAL: Questions requiring reasoning, explanation of processes, or cause-and-effect analysis
   - Examples: "How does photosynthesis work?", "Why do economies experience inflation?", "How do neural networks learn?"

3. COMPARISON: Questions asking to compare, contrast, or differentiate between multiple items
   - Examples: "What's the difference between Python and Java?", "Compare iOS vs Android", "SQL vs NoSQL databases"

4. DEFINITION: Questions asking for explanations, definitions, or clarifications of concepts
   - Examples: "Define machine learning", "Explain quantum computing", "What does API mean?"

Respond with only one word: FACTUAL, ANALYTICAL, COMPARISON, or DEFINITION""",
                ),
                ("user", "{question}"),
            ]
        )

    def _setup_response_prompts(self) -> None:
        """Set up response templates for each query type."""

        # Factual response template - concise and direct
        self.response_prompts[QueryType.FACTUAL] = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a knowledgeable assistant providing factual information. 
            
Your responses should be:
- Direct and concise
- Factually accurate
- Well-structured with clear information
- Include specific details when relevant

Format your response clearly and avoid unnecessary elaboration.""",
                ),
                ("user", "{question}"),
            ]
        )

        # Analytical response template - include reasoning steps
        self.response_prompts[QueryType.ANALYTICAL] = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an analytical assistant that explains processes and reasoning.

Your responses should:
- Break down complex topics into clear steps
- Explain the reasoning behind each step
- Use logical flow from cause to effect
- Include examples when helpful
- Structure information with numbered steps or bullet points

Format: Provide a clear explanation with step-by-step reasoning.""",
                ),
                ("user", "{question}"),
            ]
        )

        # Comparison response template - structured format
        self.response_prompts[QueryType.COMPARISON] = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a comparison specialist that helps users understand differences and similarities.

Your responses should:
- Use structured format (tables, bullet points, or clear sections)
- Compare key aspects systematically
- Highlight both similarities and differences
- Provide balanced perspective
- Include practical implications when relevant

Format: Use clear structure like:
**Similarities:**
- Point 1
- Point 2

**Differences:**
| Aspect | Item A | Item B |
|--------|--------|--------|
| Feature | Description | Description |

**Summary:** Brief conclusion about when to use each""",
                ),
                ("user", "{question}"),
            ]
        )

        # Definition response template - include examples and use cases
        self.response_prompts[QueryType.DEFINITION] = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a definition expert that provides clear explanations with practical context.

Your responses should:
- Start with a clear, concise definition
- Explain key components or characteristics
- Provide real-world examples
- Include common use cases or applications
- Use analogies when helpful for understanding

Format:
**Definition:** [Clear, concise definition]

**Key Components:**
- Component 1: Explanation
- Component 2: Explanation

**Examples:**
- Example 1 with brief explanation
- Example 2 with brief explanation

**Common Use Cases:**
- Use case 1
- Use case 2""",
                ),
                ("user", "{question}"),
            ]
        )

    def _setup_chains(self) -> None:
        """Set up the processing chains using proper chain composition."""
        if self.query_classifier_prompt is not None and self.llm is not None:
            # Classification chain
            self.classification_chain = self.query_classifier_prompt | self.llm

            # Response chains for each query type
            for query_type in QueryType:
                if query_type in self.response_prompts:
                    self.response_chains[query_type] = (
                        self.response_prompts[query_type] | self.llm
                    )

    def _classify_query(self, question: str) -> QueryType:
        """Classify the query into one of the predefined types."""
        try:
            if self.classification_chain is None:
                return QueryType.FACTUAL

            result = self.classification_chain.invoke({"question": question})
            # Handle different response types from LangChain
            if hasattr(result, "content"):
                classification = str(result.content).strip().upper()
            else:
                classification = str(result).strip().upper()

            print(f"Classification result: {classification}")

            # Map classification result to QueryType enum
            type_mapping = {
                "FACTUAL": QueryType.FACTUAL,
                "ANALYTICAL": QueryType.ANALYTICAL,
                "COMPARISON": QueryType.COMPARISON,
                "DEFINITION": QueryType.DEFINITION,
            }

            return type_mapping.get(
                classification, QueryType.FACTUAL
            )  # Default to factual
        except Exception as e:
            print(f"Classification error: {e}")
            return QueryType.FACTUAL  # Fallback to factual

    def _route_and_respond(self, question: str, query_type: QueryType) -> str:
        """Route the question to the appropriate response chain based on query type."""
        try:
            if query_type not in self.response_chains:
                return (
                    "I apologize, but I couldn't process your question type properly."
                )

            response_chain = self.response_chains[query_type]
            if response_chain is None:
                return (
                    "I apologize, but the response system is not properly configured."
                )

            result = response_chain.invoke({"question": question})
            if hasattr(result, "content"):
                return str(result.content)
            else:
                return str(result)
        except Exception as e:
            return f"I apologize, but I encountered an error processing your question: {str(e)}"

    def process_message(
        self, message: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Process a message using query understanding.

        Steps:
        1. Classify the query type
        2. Route to appropriate response template
        3. Generate formatted response

        Args:
            message: The user's input message
            chat_history: Not used in Part 1

        Returns:
            str: The assistant's response
        """
        if not self.llm:
            return "Error: Assistant not properly initialized. Please check your OpenAI API key."

        try:
            # Step 1: Classify the query
            query_type = self._classify_query(message)

            # Step 2 & 3: Route and generate response
            response = self._route_and_respond(message, query_type)

            return response

        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."
