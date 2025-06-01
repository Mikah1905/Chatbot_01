from rag_module import RAGModule
import ollama 

class Chatbot:
    def __init__(self, pdf_path):
        self.rag = RAGModule(pdf_path)
        self.model_name = "llama3"

    def ask(self, question):
        # Retrieve relevant context from PDF
        context = self.rag.retrieve_context(question)

        # Construct prompt
        prompt = f"""
Please answer the question below based on the following research paper content:

Context:
\"\"\"
{context}
\"\"\"

Question: {question}
Answer:"""

        # Query Ollama model (LLaMA3)
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        # Return answer, context, and original question
        return {
            "answer": response["message"]["content"],
            "context": context,
            "question": question
        }
