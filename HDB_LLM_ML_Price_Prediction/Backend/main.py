import pandas as pd
from fastapi import FastAPI
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import psycopg2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import requests
import os
import pickle
import json
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

app = FastAPI()

# Database connection
conn = psycopg2.connect(
    database="HDB", 
    user="postgres", 
    host='localhost',
    password="admin",
    port=5432
)
cur = conn.cursor()

# Pydantic models
class FilterRequest(BaseModel):
    town: Optional[str] = None
    flat_type: Optional[str] = None
    storey_range_classify: Optional[str] = None

class PredictionRequest(BaseModel):
    town: str
    flat_type: str
    storey_range_classify: str
    lease_remaining: float = 70.0
    floor_area_sqm: float = 75.0

class ChatRequest(BaseModel):
    message: str

# RAG System built into main.py
class SimpleRAGSystem:
    def __init__(self):
        # Azure credentials
        self.doc_endpoint = "https://govtechprocessor.cognitiveservices.azure.com/"
        self.doc_key = "98J10T3y2GmS85u2KmZOIDMU4xgIZnSngoBsx6O2U6lwzlzRJkxrJQQJ99BHACqBBLyXJ3w3AAALACOGsXr1"
        self.openai_endpoint = "https://jinsh-meh9gwl1-eastus2.openai.azure.com/"
        self.openai_key = "BvUgVIhhZg5MiyD3SppWCgWGDh8dHHUgWVCxm6ZMu4RvGZfXVdsCJQQJ99BHACHYHv6XJ3w3AAAAACOGJpZB"
        self.api_version = "2024-12-01-preview"
        
        # Initialize Azure Document Intelligence
        self.doc_client = DocumentIntelligenceClient(
            endpoint=self.doc_endpoint,
            credential=AzureKeyCredential(self.doc_key)
        )
        
        # Storage
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.data_path = "LLM/documents_to_process/azure_rag_data.pkl"  # Updated path
        
        # Load existing data or process documents
        try:
            if not self.load_data():
                print("📄 No existing RAG data found. Processing documents...")
                self.process_documents()
                if self.documents:  # Only save if we have documents
                    self.save_data()
                else:
                    print("⚠️ No documents were processed. Chatbot will use general knowledge only.")
            else:
                if len(self.documents) == 0:
                    print("⚠️ Loaded RAG data but it contains 0 documents. Reprocessing...")
                    self.process_documents()
                    if self.documents:
                        self.save_data()
                else:
                    print(f"📚 Successfully loaded RAG data with {len(self.documents)} documents")
        except Exception as e:
            print(f"⚠️ RAG initialization error: {e}")
            print("📄 Trying to process documents anyway...")
            try:
                self.process_documents()
                if self.documents:
                    self.save_data()
            except Exception as e2:
                print(f"❌ Failed to process documents: {e2}")
                # Initialize empty arrays so chatbot still works
                self.documents = []
                self.embeddings = []
                self.metadata = []
    
    def get_azure_embedding(self, text: str) -> List[float]:
        """Get embedding using Azure OpenAI"""
        try:
            url = f"{self.openai_endpoint}/openai/deployments/text-embedding-3-small/embeddings"
            
            headers = {
                "Content-Type": "application/json",
                "api-key": self.openai_key
            }
            
            data = {"input": text, "model": "text-embedding-3-small"}
            params = {"api-version": self.api_version}
            
            response = requests.post(url, headers=headers, json=data, params=params)
            
            if response.status_code == 200:
                result = response.json()
                return result["data"][0]["embedding"]
            else:
                print(f"Embedding error: {response.status_code}")
                return []
        except Exception as e:
            print(f"Embedding error: {str(e)}")
            return []
    
    def extract_text_from_document(self, file_path: str) -> str:
        """Extract text using Azure Document Intelligence"""
        try:
            with open(file_path, "rb") as f:
                document_bytes = f.read()
            
            poller = self.doc_client.begin_analyze_document("prebuilt-read", document_bytes)
            result = poller.result()
            
            return result.content if result.content else ""
        except Exception as e:
            print(f"Document processing error: {str(e)}")
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
    
    def process_documents(self):
        """Process documents from documents_to_process folder"""
        documents_folder = "LLM/documents_to_process"
        
        if not os.path.exists(documents_folder):
            print(f"❌ Documents folder '{documents_folder}' not found!")
            return
        
        files = [f for f in os.listdir(documents_folder) if os.path.isfile(os.path.join(documents_folder, f))]
        print(f"📁 Found {len(files)} files to process")
        
        all_chunks = []
        all_metadata = []
        
        for filename in files:
            file_path = os.path.join(documents_folder, filename)
            print(f"📄 Processing: {filename}")
            
            text = self.extract_text_from_document(file_path)
            if text:
                chunks = self.chunk_text(text)
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        'filename': filename,
                        'chunk_id': i,
                        'source': file_path
                    })
        
        if not all_chunks:
            print("❌ No documents processed!")
            return
        
        print(f"📊 Created {len(all_chunks)} text chunks")
        print("🔄 Creating embeddings...")
        
        # Create embeddings
        embeddings = []
        for i, chunk in enumerate(all_chunks):
            if i % 10 == 0:
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
        
        query_embedding = self.get_azure_embedding(query)
        if not query_embedding:
            return []
        
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            if doc_embedding:
                similarity = self.cosine_similarity(query_embedding, doc_embedding)
                similarities.append((similarity, i))
        
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
    
    def save_data(self):
        """Save RAG data"""
        try:
            data = {
                'documents': self.documents,
                'embeddings': self.embeddings,
                'metadata': self.metadata
            }
            
            with open(self.data_path, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"💾 RAG data saved to {self.data_path}")
        except Exception as e:
            print(f"❌ Save error: {str(e)}")
    
    def load_data(self):
        """Load existing RAG data"""
        try:
            if not os.path.exists(self.data_path):
                print(f"📝 RAG data file not found at: {self.data_path}")
                return False
            
            print(f"📂 Loading RAG data from: {self.data_path}")
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data.get('documents', [])
            self.embeddings = data.get('embeddings', [])
            self.metadata = data.get('metadata', [])
            
            print(f"📚 Loaded {len(self.documents)} documents, {len(self.embeddings)} embeddings")
            return len(self.documents) > 0  # Return True only if we have documents
        except Exception as e:
            print(f"❌ Load error: {str(e)}")
            return False

# Enhanced chatbot with RAG
class RAGChatbot:
    def __init__(self):
        self.azure_openai_endpoint = "https://jinsh-meh9gwl1-eastus2.openai.azure.com/"
        self.azure_openai_key = "BvUgVIhhZg5MiyD3SppWCgWGDh8dHHUgWVCxm6ZMu4RvGZfXVdsCJQQJ99BHACHYHv6XJ3w3AAAAACOGJpZB"
        self.api_version = "2024-12-01-preview"
        self.chat_deployment = "o4-mini"
        
        # Initialize RAG system
        print("🔧 Initializing RAG system...")
        self.rag_system = SimpleRAGSystem()
        print("✅ RAG system ready!")
    
    def azure_chat_completion(self, messages, max_tokens=1000):
        """Get chat completion using Azure OpenAI"""
        try:
            url = f"{self.azure_openai_endpoint}/openai/deployments/{self.chat_deployment}/chat/completions"
            
            headers = {
                "Content-Type": "application/json",
                "api-key": self.azure_openai_key
            }
            
            data = {
                "messages": messages,
                "max_completion_tokens": max_tokens,
                "model": self.chat_deployment
            }
            
            params = {"api-version": self.api_version}
            
            print(f"🌐 Making request to Azure OpenAI...")
            print(f"📊 Messages count: {len(messages)}")
            
            response = requests.post(url, headers=headers, json=data, params=params)
            
            print(f"📡 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"📋 Response structure: {json.dumps(result, indent=2)[:500]}...")
                
                # Extract content more carefully
                if 'choices' in result and len(result['choices']) > 0:
                    choice = result['choices'][0]
                    if 'message' in choice and 'content' in choice['message']:
                        content = choice['message']['content']
                        if content is None:
                            print("⚠️ Content is None")
                            return "I apologize, I couldn't generate a response."
                        print(f"📝 Content length: {len(str(content))} characters")
                        return str(content)
                    else:
                        print(f"❌ No content in choice: {choice}")
                        return "I couldn't generate a proper response."
                else:
                    print(f"❌ No choices in response: {result}")
                    return "No response was generated."
            else:
                error_text = response.text
                print(f"❌ API Error {response.status_code}: {error_text}")
                return f"API Error {response.status_code}: {error_text}"
                
        except Exception as e:
            print(f"❌ Exception in azure_chat_completion: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def chat(self, question):
        """Chat with RAG context"""
        print(f"🔧 Starting chat for question: {question[:50]}...")
        
        # Get relevant context from documents
        context = ""
        has_documents = len(self.rag_system.documents) > 0
        print(f"📚 Has {len(self.rag_system.documents)} documents available")
        
        if has_documents:
            try:
                context = self.rag_system.get_context_for_query(question)
                print(f"🔍 Found context: {len(context)} characters")
            except Exception as e:
                print(f"⚠️ Error getting context: {e}")
                context = ""
        else:
            print("⚠️ No RAG documents available, using general knowledge")
        
        # Simplified system prompt to reduce token usage
        system_prompt = """You are an expert HDB assistant for Singapore. Help users with HDB housing questions including resale prices, BTO applications, policies, income requirements, and procedures. Be helpful and specific."""
        
        # Build shorter user message
        if context:
            # Limit context to prevent token overflow
            context = context[:1500]  # Limit context length
            user_message = f"""Context: {context}

Question: {question}

Answer based on the context:"""
        else:
            user_message = f"""Question: {question}

Provide helpful information about HDB housing in Singapore:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        print(f"📏 Total prompt length: {len(system_prompt) + len(user_message)} characters")
        
        print("🔄 Calling Azure OpenAI...")
        try:
            # Increase max tokens significantly
            response = self.azure_chat_completion(messages, max_tokens=2000)
            print(f"✅ Got Azure response: '{response[:100]}...' (length: {len(response)})")
            
            if not has_documents and response and "Error" not in response:
                response += "\n\n*Note: This response is based on general knowledge. For current information, check the official HDB website.*"
            
            return response
        except Exception as e:
            print(f"❌ Azure chat error: {e}")
            import traceback
            traceback.print_exc()
            return f"Sorry, I'm having trouble processing your request: {str(e)}"

# Helper function
def filter_sql(
    town: Optional[str] = None,
    flat_type: Optional[str] = None,
    storey_range_classify: Optional[str] = None
):
    query = "SELECT * FROM resale_transactions WHERE true"
    values = []

    if town is not None:
        query += " AND town = %s"
        values.append(town)
    if flat_type is not None:
        query += " AND flat_type = %s"
        values.append(flat_type)
    if storey_range_classify is not None:
        query += " AND storey_range_classify = %s"
        values.append(storey_range_classify)

    cur.execute(query, tuple(values))
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=columns)

# Initialize RAG chatbot with better error handling
chatbot = None
try:
    print("🤖 Initializing RAG chatbot...")
    chatbot = RAGChatbot()
    print("✅ RAG Chatbot ready!")
except Exception as e:
    print(f"❌ RAG Chatbot initialization failed: {e}")
    print("🔄 Trying basic chatbot as fallback...")
    
    # Fallback to basic chatbot
    class BasicChatbot:
        def __init__(self):
            self.azure_openai_endpoint = "https://jinsh-meh9gwl1-eastus2.openai.azure.com/"
            self.azure_openai_key = "BvUgVIhhZg5MiyD3SppWCgWGDh8dHHUgWVCxm6ZMu4RvGZfXVdsCJQQJ99BHACHYHv6XJ3w3AAAAACOGJpZB"
            self.api_version = "2024-12-01-preview"
            self.chat_deployment = "o4-mini"
        
        def azure_chat_completion(self, messages, max_tokens=1000):
            try:
                url = f"{self.azure_openai_endpoint}/openai/deployments/{self.chat_deployment}/chat/completions"
                headers = {"Content-Type": "application/json", "api-key": self.azure_openai_key}
                data = {"messages": messages, "max_completion_tokens": max_tokens, "model": self.chat_deployment}
                params = {"api-version": self.api_version}
                
                response = requests.post(url, headers=headers, json=data, params=params)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    return f"Error: {response.status_code}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        def chat(self, question):
            system_prompt = "You are an expert HDB assistant for Singapore. Help with housing questions."
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            return self.azure_chat_completion(messages)
    
    try:
        chatbot = BasicChatbot()
        print("✅ Basic chatbot ready as fallback!")
    except Exception as e2:
        print(f"❌ All chatbot initialization failed: {e2}")
        chatbot = None

# API endpoints
@app.get("/")
async def root():
    return {"message": "HDB Data API with Chatbot"}

@app.post("/get_data")
def get_filtered_data(filters: FilterRequest):
    """Get filtered data only"""
    df = filter_sql(
        town=filters.town,
        flat_type=filters.flat_type,
        storey_range_classify=filters.storey_range_classify
    )
    
    print(f"Filtered data: {len(df)} records")
    return df.to_dict(orient='records')

@app.post("/predict")
def predict_price(request: PredictionRequest):
    """Filter data + Train model + Make prediction all in one"""
    
    # Step 1: Filter data
    df = filter_sql(
        town=request.town,
        flat_type=request.flat_type,
        storey_range_classify=request.storey_range_classify
    )
    
    if len(df) == 0:
        return {"error": "No data found for this combination"}
    
    print(f"Filtered data: {len(df)} records")
    
    # Step 2: Train model on filtered data
    try:
        # Feature engineering
        df["month"] = pd.to_datetime(df["month"])
        df["year"] = df["month"].dt.year
        current_year = 2025
        df["building_age"] = current_year - df["lease_commence_date"]
        df["lease_remaining_calc"] = 99 - df["building_age"]
        
        # Encode storey level
        storey_mapping = {'low': 0, 'medium': 1, 'high': 2}
        df['storey_level_encoded'] = df['storey_range_classify'].map(storey_mapping)
        
        # Prepare features
        X = df[['storey_level_encoded', 'lease_remaining_calc', 'floor_area_sqm']]
        y = df['resale_price']
        
        # Clean data
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) == 0:
            return {"error": "No valid training data after cleaning"}
        
        # Train model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        model = LinearRegression()
        model.fit(X_scaled, y_clean)
        
        print(f"Model trained on {len(X_clean)} records")
        
        # Step 3: Make prediction
        storey_encoded = storey_mapping[request.storey_range_classify.lower()]
        input_data = np.array([[storey_encoded, request.lease_remaining, request.floor_area_sqm]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        return {
            "predicted_price": round(prediction, 2),
            "town": request.town,
            "flat_type": request.flat_type,
            "storey_level": request.storey_range_classify,
            "lease_remaining": request.lease_remaining,
            "floor_area_sqm": request.floor_area_sqm,
            "training_records": len(X_clean)
        }
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.post("/chat")
def chat_with_bot(request: ChatRequest):
    """Chat with the HDB AI assistant"""
    print(f"📝 Received chat request: {request.message}")
    
    if chatbot is None:
        print("❌ Chatbot is None")
        return {"error": "Chatbot not available"}
    
    try:
        print(f"🤖 Calling chatbot.chat() with: {request.message}")
        response = chatbot.chat(request.message)
        print(f"✅ Got response length: {len(response)} characters")
        return {"response": response}
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        return {"error": f"Chat failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)