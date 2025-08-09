"""Part 3 - Conversation Memory implementation.

This implementation focuses on:
- Maintain context across messages
- Handle follow-up questions
- Use conversation history in responses
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
    CALCULATION = "calculation"
    FOLLOWUP = "followup"  # New type for Part 3


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

class MemoryChat(ChatInterface):
    """Week 1 Part 3 implementation adding conversation memory."""
    
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
        """Initialize components for memory-enabled chat.
        
        Sets up:
        - Chat model
        - Tool detection prompts
        - Query classification prompts (with memory awareness)
        - Response formatting prompts (with context integration)
        - Processing chains
        """
        # Initialize the chat model
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

        # Set up tool detection prompt
        self._setup_tool_detector_prompt()

        # Set up query classification prompt (memory-aware)
        self._setup_classifier_prompt()

        # Set up response formatting prompts (context-aware)
        self._setup_response_prompts()

        # Create processing chains
        self._setup_chains()

    def _setup_tool_detector_prompt(self) -> None:
        """Set up the tool detection prompt to identify when calculations are needed."""
        self.tool_detector_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a tool detector. Analyze the user's question and conversation history to determine if it requires mathematical calculation.

Look for:
- Mathematical expressions (e.g., "5 + 3", "What's 15% of 120?")
- Word problems involving numbers and operations
- Questions asking for calculations, totals, percentages, etc.
- Follow-up questions that reference previous calculations
- Any question that would benefit from using a calculator

Examples that need calculation:
- "What's 25 + 35?"
- "If I have a bill of $120, what's a 15% tip?"
- "Calculate 2.5 * 4"
- "What about 20%?" (when previous context involved percentages)
- "And for $150?" (when previous context involved bill calculations)

Examples that don't need calculation:
- "What is machine learning?"
- "How does photosynthesis work?"
- "Compare Python and Java"
- "Tell me more about that" (when previous context was not about calculations)

Consider the conversation history to understand context for follow-up questions.

Respond with only: YES (if calculation needed) or NO (if no calculation needed)""",
                ),
                ("user", "Conversation History:\n{history}\n\nCurrent Question: {question}"),
            ]
        )

    def _setup_classifier_prompt(self) -> None:
        """Set up the query classification prompt template with memory awareness."""
        self.query_classifier_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a query classifier. Analyze the user's question and conversation history to classify it into one of these categories:

1. FOLLOWUP: Questions that directly reference or build upon previous conversation context
   - Examples: "What about 20%?", "Tell me more about that", "And for Python?", "How about $150?"
   - Key indicators: pronouns (that, it), incomplete context, building on previous topics

2. CALCULATION: Questions requiring mathematical computation
   - Examples: "What's 5 + 3?", "Calculate 15% tip on $120", "What's 2.5 * 4?"

3. FACTUAL: Direct questions asking for specific facts, data, or information
   - Examples: "What is the capital of France?", "Who invented the telephone?"

4. ANALYTICAL: Questions requiring reasoning, explanation of processes
   - Examples: "How does photosynthesis work?", "Why do economies experience inflation?"

5. COMPARISON: Questions asking to compare or contrast multiple items
   - Examples: "What's the difference between Python and Java?", "Compare iOS vs Android"

6. DEFINITION: Questions asking for explanations or clarifications of concepts
   - Examples: "Define machine learning", "Explain quantum computing"

IMPORTANT: If the question seems incomplete or references previous context (like "What about...", "And for...", "Tell me more..."), classify as FOLLOWUP.

Consider the conversation history to understand the context.

Respond with only one word: FOLLOWUP, CALCULATION, FACTUAL, ANALYTICAL, COMPARISON, or DEFINITION""",
                ),
                ("user", "Conversation History:\n{history}\n\nCurrent Question: {question}"),
            ]
        )

    def _setup_response_prompts(self) -> None:
        """Set up response templates for each query type with context awareness."""

        # Follow-up response template - uses conversation history
        self.response_prompts[QueryType.FOLLOWUP] = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful assistant that excels at handling follow-up questions using conversation context.

When responding to follow-up questions:
- Use the conversation history to understand what the user is referring to
- Reference previous context naturally in your response
- If it's a calculation follow-up, extract relevant numbers/context from history
- Maintain the same topic and style as the previous conversation
- Be conversational and natural

For calculation follow-ups (like "What about 20%?" after a tip calculation):
- Extract the relevant context (bill amount, previous percentage, etc.)
- Perform the new calculation
- Reference the previous calculation for comparison

Format your response to feel natural and conversational.""",
                ),
                ("user", "Conversation History:\n{history}\n\nFollow-up Question: {question}\n\nCalculation result (if applicable): {calculation_result}"),
            ]
        )

        # Calculation response template - enhanced with context
        self.response_prompts[QueryType.CALCULATION] = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful assistant that provides clear explanations for mathematical calculations.

When presenting calculation results:
- Show the calculation clearly
- Explain what was calculated if it's a word problem
- Provide context when helpful
- If there's relevant conversation history, reference it naturally
- Keep the explanation concise but clear

Format your response to be helpful and easy to understand.""",
                ),
                ("user", "Conversation History:\n{history}\n\nQuestion: {question}\n\nCalculation result: {calculation_result}"),
            ]
        )

        # Enhanced templates with conversation history for other types
        self.response_prompts[QueryType.FACTUAL] = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a knowledgeable assistant providing factual information.

Your responses should be:
- Direct and concise
- Factually accurate
- Well-structured with clear information
- Reference conversation history if relevant
- Include specific details when relevant

If the conversation history provides relevant context, acknowledge it naturally.

Format your response clearly and avoid unnecessary elaboration.""",
                ),
                ("user", "Conversation History:\n{history}\n\nQuestion: {question}"),
            ]
        )

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
- Reference previous conversation if it adds context

Format: Provide a clear explanation with step-by-step reasoning.""",
                ),
                ("user", "Conversation History:\n{history}\n\nQuestion: {question}"),
            ]
        )

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
- Build upon previous conversation context if applicable

Format: Use clear structure with similarities, differences, and summary.""",
                ),
                ("user", "Conversation History:\n{history}\n\nQuestion: {question}"),
            ]
        )

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
- Connect to previous conversation topics if relevant

Format with definition, key components, examples, and use cases.""",
                ),
                ("user", "Conversation History:\n{history}\n\nQuestion: {question}"),
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

    def _format_history(self, chat_history: Optional[List[Dict[str, str]]]) -> str:
        """Format chat history for inclusion in prompts."""
        if not chat_history:
            return "No previous conversation."
        
        formatted = []
        for msg in chat_history[-5:]:  # Last 5 messages for context
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        
        return "\n".join(formatted) if formatted else "No previous conversation."

    def _needs_calculation(self, question: str, history: str) -> bool:
        """Detect if the question needs mathematical calculation with history context."""
        try:
            if self.tool_detection_chain is None:
                return False

            result = self.tool_detection_chain.invoke({
                "question": question,
                "history": history
            })

            # Handle different response types from LangChain
            if hasattr(result, "content"):
                detection = str(result.content).strip().upper()
            else:
                detection = str(result).strip().upper()

            return detection == "YES"
        except Exception as e:
            print(f"Tool detection error: {e}")
            return False

    def _extract_calculation_with_context(self, question: str, history: str) -> str:
        """Extract mathematical expression considering conversation context."""
        # First try to extract from current question
        expression = self._extract_calculation(question)
        
        # If question seems incomplete (like "What about 20%?"), try to use context
        if expression == question and history != "No previous conversation.":
            # Look for follow-up patterns and try to extract context
            follow_up_patterns = [
                r"what about (\d+(?:\.\d+)?%?)",  # "What about 20%?"
                r"and for \$?(\d+(?:\.\d+)?)",    # "And for $150?"
                r"how about (\d+(?:\.\d+)?%?)",   # "How about 25%?"
            ]
            
            for pattern in follow_up_patterns:
                match = re.search(pattern, question.lower())
                if match:
                    new_value = match.group(1)
                    
                    # Extract context from history
                    # Look for previous amounts, percentages, etc.
                    if "%" in new_value:
                        # Percentage follow-up - look for previous bill amount
                        bill_match = re.search(r"\$(\d+(?:\.\d+)?)", history)
                        if bill_match:
                            amount = bill_match.group(1)
                            percentage = new_value.replace('%', '')
                            return f"{float(percentage)} / 100 * {float(amount)}"
                    else:
                        # Amount follow-up - look for previous percentage
                        percent_match = re.search(r"(\d+(?:\.\d+)?)%", history)
                        if percent_match:
                            percentage = percent_match.group(1)
                            return f"{float(percentage)} / 100 * {float(new_value)}"
            
            # If no pattern matches, return the question
            return question
        
        return expression

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

    def _classify_query(self, question: str, history: str) -> QueryType:
        """Classify the query into one of the predefined types with history context."""
        try:
            if self.classification_chain is None:
                return QueryType.FACTUAL

            result = self.classification_chain.invoke({
                "question": question,
                "history": history
            })

            # Handle different response types from LangChain
            if hasattr(result, "content"):
                classification = str(result.content).strip().upper()
            else:
                classification = str(result).strip().upper()

            # Map classification result to QueryType enum
            type_mapping = {
                "FOLLOWUP": QueryType.FOLLOWUP,
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
        history: str,
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
            if query_type == QueryType.FOLLOWUP:
                input_data = {
                    "question": question,
                    "history": history,
                    "calculation_result": calculation_result or "None"
                }
            elif query_type == QueryType.CALCULATION:
                input_data = {
                    "question": question,
                    "history": history,
                    "calculation_result": calculation_result or "None"
                }
            else:
                input_data = {
                    "question": question,
                    "history": history
                }

            result = response_chain.invoke(input_data)

            # Handle different response types from LangChain
            if hasattr(result, "content"):
                return str(result.content)
            else:
                return str(result)

        except Exception as e:
            return f"I apologize, but I encountered an error processing your question: {str(e)}"
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message with memory and context awareness.
        
        Enhanced workflow with memory:
        1. Format conversation history for context
        2. Check if calculation is needed (considering context)
        3. Classify query type (including follow-up detection)
        4. Handle accordingly with appropriate context integration
        
        Args:
            message: The user's input message
            chat_history: List of previous chat messages in format [{"role": "user/assistant", "content": "..."}]
            
        Returns:
            str: The assistant's response
        """
        if not self.llm:
            return "Error: Assistant not properly initialized. Please check your OpenAI API key."

        try:
            print(f"Processing message: {message}")
            
            # Step 1: Format conversation history
            history = self._format_history(chat_history)
            print(f"Formatted history: {history}")
            
            # Step 2: Check if calculation is needed (with context)
            if self._needs_calculation(message, history):
                print("Calculation needed, extracting expression with context...")
                
                # Extract mathematical expression considering context
                expression = self._extract_calculation_with_context(message, history)
                print(f"Extracted expression: {expression}")
                
                # Use calculator tool
                calculation_result = self.calculator_tool.invoke(
                    {"expression": expression}
                )
                print(f"Calculation result: {calculation_result}")

                # Classify to determine if it's a follow-up or new calculation
                query_type = self._classify_query(message, history)
                
                # Use appropriate template based on classification
                response = self._route_and_respond(
                    message, query_type, history, calculation_result
                )
                print(f"Response after calculation: {response}")
                return response
            else:
                print("No calculation needed, classifying query...")
                
                # Step 3: Classify and respond with context awareness
                query_type = self._classify_query(message, history)
                print(f"Classified query type: {query_type}")
                
                response = self._route_and_respond(message, query_type, history)
                print(f"Response after classification: {response}")
                return response

        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."
