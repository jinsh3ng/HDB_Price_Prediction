import requests
import json
from typing import Dict, Any
from process_files import AzureRAGSystem

class SimpleChatbot:
    def __init__(self, 
                 doc_endpoint: str,
                 doc_key: str,
                 openai_endpoint: str,
                 openai_key: str):
        
        # Initialize RAG system
        self.rag = AzureRAGSystem(
            doc_intelligence_endpoint=doc_endpoint,
            doc_intelligence_key=doc_key,
            openai_endpoint=openai_endpoint,
            openai_key=openai_key,
            embedding_deployment="text-embedding-3-small",
            chat_deployment="o4-mini"
        )
        
        # Load RAG data
        if not self.rag.load_data():
            print("Processing documents...")
            self.rag.process_documents()
            self.rag.save_data()
    
    def chat(self, question: str) -> str:
        """Simple chat function"""
        # Get relevant documents
        context = self.rag.get_context_for_query(question)
        
        if not context:
            return "I don't have information about that. Please ask about HDB housing topics."
        
        # Create simple prompt
        messages = [
            {
                "role": "system", 
                "content": "You are an HDB assistant. Answer questions using the provided context."
            },
            {
                "role": "user", 
                "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            }
        ]
        
        # Get AI response
        response = self.rag.azure_chat_completion(messages)
        return response

# Usage
if __name__ == "__main__":
    # Your actual credentials
    DOC_ENDPOINT = "https://govtechprocessor.cognitiveservices.azure.com/"
    DOC_KEY = "98J10T3y2GmS85u2KmZOIDMU4xgIZnSngoBsx6O2U6lwzlzRJkxrJQQJ99BHACqBBLyXJ3w3AAALACOGsXr1"
    OPENAI_ENDPOINT = "https://jinsh-meh9gwl1-eastus2.openai.azure.com/"
    OPENAI_KEY = "BvUgVIhhZg5MiyD3SppWCgWGDh8dHHUgWVCxm6ZMu4RvGZfXVdsCJQQJ99BHACHYHv6XJ3w3AAAAACOGJpZB"
    
    # Create chatbot
    bot = SimpleChatbot(DOC_ENDPOINT, DOC_KEY, OPENAI_ENDPOINT, OPENAI_KEY)
    
    # Test
    while True:
        question = input("\nAsk me about HDB: ")
        if question.lower() in ['quit', 'exit']:
            break
        
        answer = bot.chat(question)
        print(f"\nBot: {answer}")