import ast
import asyncio
import os
from contextlib import AsyncExitStack
from typing import Optional

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm_client = OpenAI(
            api_key = os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    # methods will go here

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using LLM and available tools"""
        messages = [
            {
                "role": "user",
                "content": f"""
                {query}
                \nYou are a specialized assistant for estimating peak memory usage of scripts using the Model Context Protocol (MCP). Your goal is to provide accurate memory usage predictions by analyzing script content and characteristics.

                ## Available Tools:
                1. `get_file_content` - Retrieves and formats script content with appropriate code highlighting
                2. `get_file_size` - Provides file size information for memory baseline estimation

                ## Analysis Process:
                1. First use `get_file_content` to examine the full script
                2. Analyze the code for:
                - Data structures and their growth patterns
                - Memory-intensive operations (large matrix operations, data loading)
                - Resource allocation/deallocation patterns
                - Loops that accumulate data
                - Recursive functions and their depth
                3. Use `get_file_size` to establish a baseline for static memory requirements
                4. Consider language-specific memory characteristics (Python's overhead vs C/C++)

                ## Output Format:
                - Provide estimated peak memory range in appropriate units (KB/MB/GB)
                - Break down memory usage by major components
                - Highlight potential memory bottlenecks
                - Suggest optimization opportunities if applicable

                Remember that dynamic memory usage often exceeds static file size by orders of magnitude, especially for data processing scripts.

                """
            }
        ]

        ## STEP1: Get all file paths of files used in the script
        file_size_answer = ""
        is_answering = False

        tool_name = "get_script_contents"
        tool_args = {'file_path': "cena.py"}

        # Execute tool call
        result = await self.session.call_tool(tool_name, tool_args)

        messages.append({
            "role": "assistant",
            "content": f"""
                I'm providing you with script content via the get_file_content tool. Your task is to:
                1. Identify all relevant files that will be executed or loaded during runtime
                2. For each file, provide:
                - Filename
                - Language/runtime environment
                - Primary purpose of the file
                - Key dependencies and imports
                - Data structures being used
                - Any memory-intensive operations (e.g., large data loading, matrix operations, recursive functions)
                - Growth patterns for variables and data structures
                - Potential memory bottlenecks

                Format your response as a structured analysis with separate sections for each file. Highlight any files that appear particularly memory-intensive.

                For Python files, pay special attention to:
                - NumPy/Pandas/TensorFlow operations
                - Large data loading operations
                - List/dictionary comprehensions
                - Unbounded append operations in loops
                - Recursive functions without clear termination conditions

                For JavaScript files, focus on:
                - Large array operations
                - DOM manipulation
                - Memory leaks from closures
                - Event listeners that might prevent garbage collection

                For compiled languages (C/C++/Rust), note:
                - Manual memory allocation/deallocation
                - Potential memory leaks
                - Large static allocations

                List all files that should be analyzed with the get_file_size tool in the next step.
                \nHere are the contents of the script:
                \n{result.content}
                \nPlease get all file paths of files used in the script and collect them into a list
                \nOnly give me the list of paths. Do not include any other information
            """
        })
        response = self.llm_client.chat.completions.create(
            model="deepseek-r1",
            messages=messages,
            stream=True,
            # stream_options={
            #     "include_usage": True
            # }
        )
        for chunk in response:
            if not chunk.choices:
                print("\nUsage:")
                print(chunk.usage)
            else:
                delta = chunk.choices[0].delta
                # print reasoning content
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                    print(delta.reasoning_content, end='', flush=True)
                else:
                    # answer
                    if delta.content != "" and is_answering is False:
                        print("\n" + "=" * 20 + "Answer" + "=" * 20 + "\n")
                        is_answering = True
                    print(delta.content, end='', flush=True)
                    file_size_answer += delta.content


        file_sizes = {}
        tool_name = "get_file_size"
        for file in ast.literal_eval(file_size_answer):
            tool_args = {'file_path': file}

            # Execute tool call
            result = await self.session.call_tool(tool_name, tool_args)
            file_sizes[file] = result.content

        ## STEP2: Infer the memory usage from the file infos and estimate the memory usage of the script
        messages.append({
            "role": "assistant",
            "content": f"""
                Based on the file analysis from the previous step, I've provided file size information using the get_file_size tool. Now, estimate the peak memory usage during script execution by:

                1. Starting with baseline memory requirements:
                - Runtime environment overhead (Python interpreter: ~15-30MB, Node.js: ~40-60MB, etc.)
                - Static code size (from get_file_size results)
                - Imported libraries' typical memory footprint

                2. Analyzing dynamic memory patterns:
                - Estimate size of data structures at their peak
                - Calculate memory for operations like matrix multiplication (rows × columns × data type size)
                - Account for temporary variables and intermediate results
                - Consider garbage collection cycles and memory retention

                3. Provide a comprehensive estimate with:
                - Minimum memory requirement
                - Expected peak memory usage
                - Worst-case scenario memory usage
                - Breakdown by major components (e.g., "Data loading: ~500MB, Model training: ~2GB")
                - Confidence level in your estimation (high/medium/low)
                - Key factors that could significantly increase memory usage

                4. Optimization recommendations:
                - Specific code sections that could be optimized
                - Alternative approaches for memory-intensive operations
                - Potential for chunking or streaming data

                Present your analysis in a clear, structured format with quantitative estimates where possible. If there's significant uncertainty in any area, explain the factors contributing to that uncertainty.
                \nHere are the file sizes of the files used in the script:
                \n{file_sizes}
                \nInfer the memory usage from these file infos and estimate the memory usage of the script
                \nAnd give me your most confident estimation
            """
        })

        is_answering = False
        answer_content = ""

        response = self.llm_client.chat.completions.create(
            model="deepseek-r1",
            messages=messages,
            stream=True,
            # stream_options={
            #     "include_usage": True
            # }
        )
        for chunk in response:
            if not chunk.choices:
                print("\nUsage:")
                print(chunk.usage)
            else:
                delta = chunk.choices[0].delta
                # print reasoning content
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                    print(delta.reasoning_content, end='', flush=True)
                else:
                    # answer
                    if delta.content != "" and is_answering is False:
                        print("\n" + "=" * 20 + "Answer" + "=" * 20 + "\n")
                        is_answering = True
                    print(delta.content, end='', flush=True)
                    answer_content += delta.content


        return answer_content

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())