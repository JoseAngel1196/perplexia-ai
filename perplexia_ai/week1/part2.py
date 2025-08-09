"""Part 2 - Basic Tools implementation.

This implementation focuses on:
- Detect when calculations are needed
- Use calculator for mathematical operations
- Format calculation results clearly
"""

import re
from typing import Dict, List, Optional, Any
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator


class QueryType(Enum):
    """Enumeration of different query types for classification."""

    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARISON = "comparison"
    DEFINITION = "definition"
    CALCULATION = "calculation"  # New type for Part 2


# Create calculator tool using LangChain's @tool decorator
@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations.

    Args:
        expression: A mathematical expression to evaluate (e.g., "5 + 3", "10 * (2 + 3)")

    Returns:
        str: The result of the calculation or an error message
    """
    calc = Calculator()
    result = calc.evaluate_expression(expression)

    if isinstance(result, str) and result.startswith("Error"):
        return result
    else:
        return f"The result is: {result}"


class BasicToolsChat(ChatInterface):
    """Week 1 Part 2 implementation adding calculator functionality."""

    def __init__(self):
        self.llm: Optional[ChatOpenAI] = None
        self.tool_detector_prompt: Optional[ChatPromptTemplate] = None
        self.query_classifier_prompt: Optional[ChatPromptTemplate] = None
        self.response_prompts: Dict[QueryType, ChatPromptTemplate] = {}
        self.calculator_tool = calculate
        self.tool_detection_chain = None
        self.classification_chain = None
        self.response_chains: Dict[QueryType, Any] = {}

    def initialize(self) -> None:
        """Initialize components for basic tools.

        Sets up:
        - Chat model
        - Tool detection prompts
        - Query classification prompts
        - Response formatting prompts
        - Processing chains
        """
        # Initialize the chat model
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

        # Set up tool detection prompt
        self._setup_tool_detector_prompt()

        # Set up query classification prompt
        self._setup_classifier_prompt()

        # Set up response formatting prompts
        self._setup_response_prompts()

        # Create processing chains
        self._setup_chains()

    def _setup_tool_detector_prompt(self) -> None:
        """Set up the tool detection prompt to identify when calculations are needed."""
        self.tool_detector_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a tool detector. Analyze the user's question and determine if it requires mathematical calculation.

Look for:
- Mathematical expressions (e.g., "5 + 3", "What's 15% of 120?")
- Word problems involving numbers and operations
- Questions asking for calculations, totals, percentages, etc.
- Any question that would benefit from using a calculator

Examples that need calculation:
- "What's 25 + 35?"
- "If I have a bill of $120, what's a 15% tip?"
- "Calculate 2.5 * 4"
- "What's the square of 7?"

Examples that don't need calculation:
- "What is machine learning?"
- "How does photosynthesis work?"
- "Compare Python and Java"

Respond with only: YES (if calculation needed) or NO (if no calculation needed)""",
                ),
                ("user", "{question}"),
            ]
        )

    def _setup_classifier_prompt(self) -> None:
        """Set up the query classification prompt template (extended from Part 1)."""
        self.query_classifier_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a query classifier. Analyze the user's question and classify it into one of these categories:

1. CALCULATION: Questions requiring mathematical computation
   - Examples: "What's 5 + 3?", "Calculate 15% tip on $120", "What's 2.5 * 4?"

2. FACTUAL: Direct questions asking for specific facts, data, or information
   - Examples: "What is the capital of France?", "Who invented the telephone?"

3. ANALYTICAL: Questions requiring reasoning, explanation of processes
   - Examples: "How does photosynthesis work?", "Why do economies experience inflation?"

4. COMPARISON: Questions asking to compare or contrast multiple items
   - Examples: "What's the difference between Python and Java?", "Compare iOS vs Android"

5. DEFINITION: Questions asking for explanations or clarifications of concepts
   - Examples: "Define machine learning", "Explain quantum computing"

Respond with only one word: CALCULATION, FACTUAL, ANALYTICAL, COMPARISON, or DEFINITION""",
                ),
                ("user", "{question}"),
            ]
        )

    def _setup_response_prompts(self) -> None:
        """Set up response templates for each query type (extended from Part 1)."""

        # Calculation response template - clear and explanatory
        self.response_prompts[QueryType.CALCULATION] = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful assistant that provides clear explanations for mathematical calculations.

When presenting calculation results:
- Show the calculation clearly
- Explain what was calculated if it's a word problem
- Provide context when helpful
- Keep the explanation concise but clear

Format your response to be helpful and easy to understand.""",
                ),
                ("user", "{question}\n\nCalculation result: {calculation_result}"),
            ]
        )

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
        """Set up the processing chains."""
        if self.llm is not None:
            # Tool detection chain
            if self.tool_detector_prompt is not None:
                self.tool_detection_chain = self.tool_detector_prompt | self.llm

            # Classification chain
            if self.query_classifier_prompt is not None:
                self.classification_chain = self.query_classifier_prompt | self.llm

            # Response chains for each query type
            for query_type in QueryType:
                if query_type in self.response_prompts:
                    self.response_chains[query_type] = (
                        self.response_prompts[query_type] | self.llm
                    )

    def _needs_calculation(self, question: str) -> bool:
        """Detect if the question needs mathematical calculation."""
        try:
            if self.tool_detection_chain is None:
                return False

            result = self.tool_detection_chain.invoke({"question": question})

            # Handle different response types from LangChain
            if hasattr(result, "content"):
                detection = str(result.content).strip().upper()
            else:
                detection = str(result).strip().upper()

            return detection == "YES"
        except Exception as e:
            print(f"Tool detection error: {e}")
            return False

    def _extract_calculation(self, question: str) -> str:
        """Extract mathematical expression from the question."""
        # Simple regex patterns to extract mathematical expressions
        patterns = [
            r"(?:calculate|what\'s|whats)\s+([0-9+\-*/().\s]+)",
            r"([0-9+\-*/().\s]+)\s*[=?]",
            r"(\d+(?:\.\d+)?\s*[+\-*/]\s*\d+(?:\.\d+)?(?:\s*[+\-*/]\s*\d+(?:\.\d+)?)*)",
            # Handle percentage calculations
            r"(\d+(?:\.\d+)?)\s*%\s*(?:of|tip on)\s*\$?(\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:  # Percentage case
                    percentage, amount = match.groups()
                    return f"{float(percentage)} / 100 * {float(amount)}"
                else:
                    return match.group(1).strip()

        # If no pattern matches, return the question (let the calculator handle it)
        return question

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

            # Map classification result to QueryType enum
            type_mapping = {
                "CALCULATION": QueryType.CALCULATION,
                "FACTUAL": QueryType.FACTUAL,
                "ANALYTICAL": QueryType.ANALYTICAL,
                "COMPARISON": QueryType.COMPARISON,
                "DEFINITION": QueryType.DEFINITION,
            }

            return type_mapping.get(classification, QueryType.FACTUAL)
        except Exception as e:
            print(f"Classification error: {e}")
            return QueryType.FACTUAL

    def _route_and_respond(
        self,
        question: str,
        query_type: QueryType,
        calculation_result: Optional[str] = None,
    ) -> str:
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

            # Prepare input based on query type
            if query_type == QueryType.CALCULATION and calculation_result:
                input_data = {
                    "question": question,
                    "calculation_result": calculation_result,
                }
            else:
                input_data = {"question": question}

            result = response_chain.invoke(input_data)

            # Handle different response types from LangChain
            if hasattr(result, "content"):
                return str(result.content)
            else:
                return str(result)

        except Exception as e:
            return f"I apologize, but I encountered an error processing your question: {str(e)}"

    def process_message(
        self, message: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Process a message with calculator support.

        Enhanced workflow:
        1. Check if calculation is needed
        2. If yes, use calculator tool and format result
        3. If no, classify query type and respond normally

        Args:
            message: The user's input message
            chat_history: Not used in Part 2

        Returns:
            str: The assistant's response
        """
        if not self.llm:
            return "Error: Assistant not properly initialized. Please check your OpenAI API key."

        try:
            print(f"Processing message: {message}")
            # Step 1: Check if calculation is needed
            if self._needs_calculation(message):
                print("Calculation needed, extracting expression...")
                # Extract mathematical expression
                expression = self._extract_calculation(message)
                print(f"Extracted expression: {expression}")
                # Use calculator tool
                calculation_result = self.calculator_tool.invoke(
                    {"expression": expression}
                )
                print(f"Calculation result: {calculation_result}")

                # Format response using calculation template
                response = self._route_and_respond(
                    message, QueryType.CALCULATION, calculation_result
                )
                print(f"Response after calculation: {response}")
                return response
            else:
                print("No calculation needed, classifying query...")
                # Step 2: Classify and respond normally (like Part 1)
                query_type = self._classify_query(message)
                print(f"Classified query type: {query_type}")
                response = self._route_and_respond(message, query_type)
                print(f"Response after classification: {response}")
                return response

        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."
