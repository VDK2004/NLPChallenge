import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import os
from duckduckgo_search import DDGS
import trafilatura
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

class WebSearcher:
    def __init__(self):
        self.ddg = DDGS()
        self.executor = ThreadPoolExecutor(max_workers=5)
        
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Search the web for information using DuckDuckGo
        
        Args:
            query (str): Search query
            num_results (int): Number of results to return
            
        Returns:
            List[Dict]: List of search results with content and metadata
        """
        try:
            # Perform DuckDuckGo search
            search_results = []
            ddg_results = list(self.ddg.text(query, max_results=num_results))
            
            # Extract content from webpages in parallel
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            futures = []
            
            for result in ddg_results:
                futures.append(
                    loop.run_in_executor(
                        self.executor,
                        self._extract_content,
                        result['link']
                    )
                )
            
            # Gather results
            contents = loop.run_until_complete(asyncio.gather(*futures))
            loop.close()
            
            # Combine search results with extracted content
            for result, content in zip(ddg_results, contents):
                if content:
                    search_results.append({
                        "title": result['title'],
                        "url": result['link'],
                        "snippet": result['body'],
                        "content": content
                    })
            
            return search_results
            
        except Exception as e:
            print(f"Error performing web search: {str(e)}")
            return []
    
    def _extract_content(self, url: str) -> str:
        """
        Extract main content from a webpage
        
        Args:
            url (str): URL to extract content from
            
        Returns:
            str: Extracted content
        """
        try:
            # Download webpage
            downloaded = trafilatura.fetch_url(url)
            
            # Extract main content
            if downloaded:
                content = trafilatura.extract(
                    downloaded,
                    include_links=True,
                    include_images=False,
                    include_tables=False,
                    no_fallback=True
                )
                return content if content else ""
            return ""
        except Exception as e:
            print(f"Error extracting content: {str(e)}")
            return ""
