import os
import traceback
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from rag_app.settings import BASE_DIR

class RAGChatService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
        )
        
        self.vector_store = Chroma(
            collection_name="rag_app_collection",
            embedding_function=self.embeddings,
            persist_directory=os.path.join(BASE_DIR, 'chroma_langchain_db'),
        )
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            
        )
        
        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
    def format_docs(self, docs):
        """Format retrieved documents into a readable context string"""
        return "\n\n".join([
            f"Document {i+1}:\n{doc.page_content}" 
            for i, doc in enumerate(docs)
        ])
    
    def get_chat_history_messages(self, chat_history: List[Dict[str, str]]):
        """Convert chat history to LangChain message format"""
        messages = []
        for chat in chat_history:
            messages.append(HumanMessage(content=chat['user_message']))
            messages.append(AIMessage(content=chat['ai_message']))
        return messages
    
    def create_rag_chain(self):
        """Create the RAG chain with chat history support"""
        
        # System prompt for RAG
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context from uploaded documents.

        Use the following context to answer the user's question. If you cannot find the answer in the context, say so honestly - don't make up information.

        Context from documents:
        {context}

        Instructions:
        - Answer based primarily on the provided context
        - Be concise but informative
        - If the context doesn't contain enough information, acknowledge this
        - You can use your general knowledge to supplement, but clearly distinguish this
        - Cite specific parts of documents when relevant"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{question}")
        ])
        
        # Create the RAG chain with proper input handling
        def get_question(x):
            """Extract question string from input"""
            if isinstance(x, dict):
                return x.get("question", "")
            return str(x)
        
        def get_history(x):
            """Extract chat history from input"""
            if isinstance(x, dict):
                return x.get("chat_history", [])
            return []
        
        # Create retrieval chain that only uses the question string
        retrieval_chain = (
            RunnablePassthrough.assign(
                context=lambda x: self.format_docs(
                    self.retriever.invoke(get_question(x))
                )
            )
        )
        
        # Complete RAG chain
        rag_chain = (
            retrieval_chain | prompt | self.llm | StrOutputParser()
        )
        
        return rag_chain
    
    def get_response(
        self, 
        question: str, 
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Get response from RAG system
        
        Args:
            question: User's question
            chat_history: List of previous exchanges [{"user_message": "...", "ai_message": "..."}]
            
        Returns:
            Dict with 'answer' and 'source_documents'
        """
        try:
            # Get relevant documents using just the question string
            # source_docs = self.retriever.invoke(question)
            
            # Format chat history if provided
            formatted_history = []
            if chat_history:
                formatted_history = self.get_chat_history_messages(chat_history)
            
            # Create and invoke RAG chain
            rag_chain = self.create_rag_chain()
            
            # Invoke with properly structured input
            answer = rag_chain.invoke({
                "question": question,
                "chat_history": formatted_history
            })
            
            return {
                "answer": answer,
                # "source_documents": [
                #     {
                #         "content": doc.page_content,
                #         "metadata": doc.metadata
                #     }
                #     for doc in source_docs
                # ]
            }
            
        except Exception as e:
            import traceback
            print(f"Error in get_response: {str(e)}")
            print(traceback.format_exc())
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "source_documents": []
            }
    
    def get_relevant_documents(self, query: str, k: int = 4) -> List[Dict]:
        """Get relevant documents without generating a response"""
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
        except Exception as e:
            print(f"Error in get_relevant_documents: {str(e)}")
            return []
