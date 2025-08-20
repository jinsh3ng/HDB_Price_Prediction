import os
import numpy as np
import pickle
import json
from typing import List, Dict, Any
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
import requests

class AzureRAGSystem:
    def __init__(self, 
                 doc_intelligence_endpoint: str,
                 doc_intelligence_key: str,
                 openai_endpoint: str,
                 openai_key: str,
                 embedding_deployment: str = "text-embedding-3-small",
                 chat_deployment: str = "gpt-4",
                 documents_folder: str = "documents_to_process"):
        
        # Document Intelligence setup
        self.doc_endpoint = doc_intelligence_endpoint
        self.doc_key = doc_intelligence_key
        self.doc_client = DocumentIntelligenceClient(
            endpoint=doc_intelligence_endpoint,
            credential=AzureKeyCredential(doc_intelligence_key)
        )
        
        # Azure OpenAI setup
        self.openai_endpoint = openai_endpoint.rstrip('/')
        self.openai_key = openai_key
        self.embedding_deployment = embedding_deployment
        self.chat_deployment = chat_deployment
        self.api_version = "2024-12-01-preview"
        
        # Documents folder
        self.documents_folder = documents_folder
        
        # Storage
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.data_path = "azure_rag_data.pkl"
    
    def extract_text_from_document(self, file_path: str) -> str:
        """Extract text using Azure Document Intelligence"""
        try:
            print(f"  📄 Processing: {os.path.basename(file_path)}")
            
            with open(file_path, "rb") as f:
                document_bytes = f.read()
            
            # Analyze document
            poller = self.doc_client.begin_analyze_document(
                "prebuilt-read", 
                document_bytes
            )
            result = poller.result()
            
            text_content = ""
            if result.content:
                text_content = result.content
            
            print(f"     ✅ Extracted {len(text_content)} characters")
            return text_content
            
        except Exception as e:
            print(f"     ❌ Error: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    chunk = text[start:break_point + 1]
                    end = break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def get_azure_embedding(self, text: str) -> List[float]:
        """Get embedding using Azure OpenAI"""
        try:
            url = f"{self.openai_endpoint}/openai/deployments/{self.embedding_deployment}/embeddings"
            
            headers = {
                "Content-Type": "application/json",
                "api-key": self.openai_key
            }
            
            data = {
                "input": text,
                "model": self.embedding_deployment
            }
            
            params = {"api-version": self.api_version}
            
            response = requests.post(url, headers=headers, json=data, params=params)
            
            if response.status_code == 200:
                result = response.json()
                return result["data"][0]["embedding"]
            else:
                print(f"     ❌ Embedding error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            print(f"     ❌ Embedding error: {str(e)}")
            return []
    
    def azure_chat_completion(self, messages: List[Dict], max_tokens: int = 1000) -> str:
        """Get chat completion using Azure OpenAI"""
        try:
            url = f"{self.openai_endpoint}/openai/deployments/{self.chat_deployment}/chat/completions"
            
            headers = {
                "Content-Type": "application/json",
                "api-key": self.openai_key
            }
            
            data = {
                "messages": messages,
                "max_completion_tokens": max_tokens,
                "model": self.chat_deployment
            }
            
            params = {"api-version": self.api_version}
            
            response = requests.post(url, headers=headers, json=data, params=params)
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"❌ Chat error: {response.status_code} - {response.text}")
                return f"Error: {response.status_code}"
                
        except Exception as e:
            print(f"❌ Chat error: {str(e)}")
            return f"Error: {str(e)}"
    
    def process_documents(self):
        """Process all documents"""
        print("🏠 Azure HDB RAG System - Processing Documents")
        print("=" * 60)
        
        if not os.path.exists(self.documents_folder):
            print(f"❌ Folder '{self.documents_folder}' not found!")
            return
        
        files = [f for f in os.listdir(self.documents_folder) 
                if os.path.isfile(os.path.join(self.documents_folder, f))]
        
        print(f"📁 Found {len(files)} files:")
        for file in files:
            print(f"   - {file}")
        
        if not files:
            print("❌ No files found!")
            return
        
        all_chunks = []
        all_metadata = []
        
        # Process each document
        for filename in files:
            file_path = os.path.join(self.documents_folder, filename)
            
            # Extract text with Azure Document Intelligence
            text = self.extract_text_from_document(file_path)
            
            if text:
                # Create chunks
                chunks = self.chunk_text(text)
                print(f"     📝 Created {len(chunks)} chunks")
                
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        'filename': filename,
                        'chunk_id': i,
                        'source': file_path
                    })
        
        if not all_chunks:
            print("❌ No text extracted!")
            return
        
        print(f"\n✅ Total: {len(all_chunks)} chunks from {len(files)} documents")
        
        # Create embeddings with Azure OpenAI
        print("\n🔄 Creating embeddings with Azure OpenAI...")
        embeddings = []
        
        for i, chunk in enumerate(all_chunks):
            if i % 5 == 0:
                print(f"   Progress: {i+1}/{len(all_chunks)}")
            
            embedding = self.get_azure_embedding(chunk)
            embeddings.append(embedding)
        
        self.documents = all_chunks
        self.embeddings = embeddings
        self.metadata = all_metadata
        
        print(f"✅ Created {len(embeddings)} embeddings")
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except:
            return 0.0
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        if not self.documents or not self.embeddings:
            return []
        
        # Get query embedding
        query_embedding = self.get_azure_embedding(query)
        if not query_embedding:
            return []
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            if doc_embedding:
                similarity = self.cosine_similarity(query_embedding, doc_embedding)
                similarities.append((similarity, i))
        
        # Sort and get top k
        similarities.sort(reverse=True)
        top_results = similarities[:k]
        
        results = []
        for similarity, idx in top_results:
            results.append({
                'content': self.documents[idx],
                'metadata': self.metadata[idx],
                'score': similarity
            })
        
        return results
    
    def get_context_for_query(self, query: str, max_context_length: int = 3000) -> str:
        """Get relevant context for a query"""
        results = self.search(query, k=5)
        
        context_parts = []
        current_length = 0
        
        for result in results:
            content = result['content']
            if current_length + len(content) <= max_context_length:
                context_parts.append(f"Source: {result['metadata']['filename']}\n{content}")
                current_length += len(content)
            else:
                remaining_space = max_context_length - current_length
                if remaining_space > 100:
                    truncated = content[:remaining_space-10] + "..."
                    context_parts.append(f"Source: {result['metadata']['filename']}\n{truncated}")
                break
        
        return "\n\n---\n\n".join(context_parts)
    
    def chat_with_context(self, user_query: str) -> str:
        """Chat with RAG context"""
        # Get relevant context
        context = self.get_context_for_query(user_query)
        
        # Build messages
        system_prompt = """You are an expert HDB (Housing Development Board) assistant for Singapore. 
You help users with HDB resale prices, BTO applications, housing policies, and market insights.
Use the provided context to give accurate, specific answers with numbers when possible."""
        
        user_message = f"""Context Information:
{context}

User Question: {user_query}

Please provide a comprehensive answer using the context above. Reference specific sources when mentioning facts or data."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        return self.azure_chat_completion(messages)
    
    def save_data(self):
        """Save all data"""
        try:
            data = {
                'documents': self.documents,
                'embeddings': self.embeddings,
                'metadata': self.metadata
            }
            
            with open(self.data_path, 'wb') as f:
                pickle.dump(data, f)
            
            file_size = os.path.getsize(self.data_path) / 1024 / 1024
            print(f"✅ Data saved to {self.data_path} ({file_size:.1f} MB)")
            
        except Exception as e:
            print(f"❌ Save error: {str(e)}")
    
    def load_data(self):
        """Load saved data"""
        try:
            if not os.path.exists(self.data_path):
                print("📝 No existing data found")
                return False
            
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data['documents']
            self.embeddings = data['embeddings']
            self.metadata = data['metadata']
            
            print(f"✅ Loaded {len(self.documents)} documents")
            return True
            
        except Exception as e:
            print(f"❌ Load error: {str(e)}")
            return False

def test_system(rag: AzureRAGSystem):
    """Test the system"""
    test_queries = [
        "What are the income requirements for BTO flats?",
        "How to apply for HDB resale flat?",
        "What is the price range for 4-room flats?",
        "BTO application process steps"
    ]
    
    print("\n" + "=" * 60)
    print("🔍 TESTING SEARCH & CHAT")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        
        # Test search
        results = rag.search(query, k=3)
        if results:
            print(f"📊 Found {len(results)} relevant documents:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result['score']:.3f} | {result['metadata']['filename']}")
        
        # Test chat
        print("🤖 AI Response:")
        response = rag.chat_with_context(query)
        print(f"   {response[:200]}...")
        print()

if __name__ == "__main__":
    print("🏠 Azure-Only HDB RAG System")
    print("Azure Document Intelligence + Azure OpenAI")
    print("=" * 60)
    
    # YOUR AZURE CREDENTIALS - UPDATE THESE
    DOC_INTELLIGENCE_ENDPOINT = "https://govtechprocessor.cognitiveservices.azure.com/"
    DOC_INTELLIGENCE_KEY = "98J10T3y2GmS85u2KmZOIDMU4xgIZnSngoBsx6O2U6lwzlzRJkxrJQQJ99BHACqBBLyXJ3w3AAALACOGsXr1"
    
    # ADD YOUR AZURE OPENAI CREDENTIALS
    AZURE_OPENAI_ENDPOINT = "https://jinsh-meh9gwl1-eastus2.openai.azure.com/" 
    AZURE_OPENAI_KEY = "BvUgVIhhZg5MiyD3SppWCgWGDh8dHHUgWVCxm6ZMu4RvGZfXVdsCJQQJ99BHACHYHv6XJ3w3AAAAACOGJpZB" 
    EMBEDDING_DEPLOYMENT = "text-embedding-3-small"  
    CHAT_DEPLOYMENT = "o4-mini"  
    
    # Initialize system
    rag = AzureRAGSystem(
        doc_intelligence_endpoint=DOC_INTELLIGENCE_ENDPOINT,
        doc_intelligence_key=DOC_INTELLIGENCE_KEY,
        openai_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_key=AZURE_OPENAI_KEY,
        embedding_deployment=EMBEDDING_DEPLOYMENT,
        chat_deployment=CHAT_DEPLOYMENT
    )
    
    # Process or load data
    if not rag.load_data():
        print("Processing new documents...")
        rag.process_documents()
        if rag.documents:
            rag.save_data()
    
    # Test the system
    if rag.documents:
        test_system(rag)
        print("\n🎉 Azure RAG system ready!")
    else:
        print("❌ No documents processed!")