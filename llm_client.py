from __future__ import annotations

import enum
import json
import re
from typing import Any

import lmstudio as lms
import requests
from ddgs import DDGS


class InternetAccessLLM:
    def __init__(self, prompt_file: str = "prompt.txt", model: str = "qwen2.5-7b-instruct"):
        self.response = None
        self.model_name = model
        self.prompt_file = prompt_file
        self.system_prompt = self._load_prompt()

        self.web_text_limit = 10000
        self.search_links_limit = 10

    def _load_prompt(self) -> str:
        """ Load the system prompt from prompt.txt """
        with open(self.prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()

    def web_search(self, query: str, max_results: int = 5) -> str:
        """Search the internet using DuckDuckGo. Use this to search the internet. Returns a list of search results with snippets of text."""
        try:
            with DDGS() as ddgs:
                results = []
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    })

                return json.dumps({
                    "query": query,
                    "results": results,
                    "count": len(results)
                })
        except Exception as e:
            return json.dumps({"error": f"Search error: {str(e)}"})

    def visit_website(self, url: str, extract_type: str = "text") -> str:
        """Visit a specific website URL and retrieve its content. Use this to access the specific page found by the search engine.

        Args:
            url: The URL of the website to visit and extract content from
            extract_type: Type of content to extract: 'text' for main text content, 'links' for all links, 'metadata' for page metadata
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=15)

            if response.status_code == 200:
                content = response.text

                if extract_type == "text":
                    # Remove HTML tags for basic text extraction
                    text = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
                    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
                    text = re.sub(r'<[^>]+>', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()

                    # Limit length
                    text = text[:self.web_text_limit]

                    return json.dumps({
                        "url": url,
                        "type": "text",
                        "content": text
                    })

                elif extract_type == "links":
                    links = re.findall(r'href=["\']([^"\']+)["\']', content)
                    return json.dumps({
                        "url": url,
                        "type": "links",
                        "links": links[:self.search_links_limit]  # Limit links number
                    })

                elif extract_type == "metadata":
                    title_match = re.search(r'<title>([^<]+)</title>', content)
                    title = title_match.group(1) if title_match else ""

                    return json.dumps({
                        "url": url,
                        "type": "metadata",
                        "title": title,
                        "status": response.status_code
                    })

            else:
                return json.dumps({"error": f"Failed to visit {url}, status: {response.status_code}"})

        except Exception as e:
            return json.dumps({"error": f"Error visiting {url}: {str(e)}"})

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        """Extract JSON object from text response"""
        # First, try to parse the entire response and find JSON object
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON within code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON object
        json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # No JSON found
        return None

    def _process_response_stream(self, prediction_stream, verbose: bool = True) -> str:
        """Process streaming response from LM Studio"""
        final_content = ""

        for fragment in prediction_stream:
            content = fragment.content

            if content and verbose:
                print(content, end="", flush=True)

            final_content += content

        if verbose:
            print("\n")

        return final_content

    def run_rest_api(self, user_prompt: str) -> dict | None:
        """
            Use a scoped model handle so LM Studio tears down the connection
            (and its server-side KV state) after every single call.
            """
        try:
            with lms.Client() as client:
                model = client.llm.model(self.model_name)
                chat = lms.Chat(self.system_prompt)
                chat.add_user_message(user_prompt)
                result = model.respond(chat, config={
                    "temperature": 0,
                    "responseFormat": {"type": "json_object"},
                })
                response_text = result.content if hasattr(result, "content") else str(result)
                return self._extract_json(response_text)
        except Exception as e:
            print(f"❌ LLM call failed: {e}")
            return None

    def run(self, user_prompt: str, internet_access: bool = True,
            force_json: bool = False, verbose: bool = True, max_tool_calls=3):
        """
        Run the LLM with optional internet access using LM Studio's .act() API
        """
        try:
            # Recreate the client for each run
            self.client = lms.llm(self.model_name)
            self.client.contextOverflowPolicy = 'truncateMiddle'

            # Create new chat with system prompt
            chat = lms.Chat(self.system_prompt)
            chat.add_user_message(user_prompt)

            # Define tools for internet access
            tools = [self.web_search, self.visit_website] if internet_access else []

            # Configuration for the model
            config = {
                "temperature": 0,
                "maxToolRoundtrips": max_tool_calls,
            }

            if force_json:
                config["responseFormat"] = {"type": "json_object"}

            def print_fragment(fragment, round_index=None):
                """Callback for printing fragments during generation"""
                if verbose:
                    print(fragment.content, end="", flush=True)

            def print_message(message, round_index=None):
                """Callback for printing fragments during generation"""
                if verbose and 'TextData' not in message.content:
                    print(message.content)

            def on_prediction_completed(final_result, round_index=None):
                """
                Callback called when the LLM finishes generation.
                Cleans up the final result for printing or further processing.
                """
                # Get the main assistant output
                self.response = getattr(final_result, "content", str(final_result))

            if internet_access and tools:
                # Use tool calling
                result = self.client.act(
                    chat,
                    tools,
                    config=config,
                    on_message=print_message if verbose else None,
                    on_prediction_fragment=print_fragment if verbose else None,
                    on_prediction_completed=on_prediction_completed
                )

                if verbose:
                    print("\n")

                # Get the final response
                response_text = self.response
            else:
                # Use .respond() for non-tool queries
                if verbose:
                    prediction_stream = self.client.respond_stream(chat, config=config)
                    response_text = self._process_response_stream(prediction_stream, verbose=verbose)
                else:
                    result = self.client.respond(chat, config=config)
                    response_text = result.content if hasattr(result, 'content') else str(result)

            # Extract JSON if requested
            if force_json:
                return self._extract_json(response_text)

            return response_text

        except Exception as e:
            print(f"❌ LLM prompt failed: {e}")
            return None