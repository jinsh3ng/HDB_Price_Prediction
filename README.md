# GovTech Take-home Assignment 
Hi recruiter, thank you for reviewing my application and giving me the opportunity to showcase my work. 

To be upfront, this week was unusually busy for me, and I slightly exceeded the 36-hour time frame provided. Despite that, I was able to build an application that minimally captures the core functionality of the task, and I hope it meets your expectations. 

I also want to acknowledge that I used AI tools as part of my development process. However, the architecture, and implementation logic were driven by me.<table>

## System Architecture
Apologies if anything is confusing, this is my first time drawing such a diagram.
<img width="683" height="550" alt="SystemArchitecture" src="https://github.com/user-attachments/assets/40af03a9-de75-466c-af76-6f3e20a3fae4" />

---
## Demostration


https://github.com/user-attachments/assets/fdef8382-6cd3-44d8-a412-4ae8c2d24e51

---

## 📁 Repository Structure

This repository contains the following main folders:

### `/data_cleaning`
- Combines and cleans the original datasets from HDB.gov.
- The cleaned data is then manually imported into a PostgreSQL database.

### `/data_analysis`
- Performs basic analysis to address the question:  
  _"Please recommend housing estates that have had limited Build-To-Order (BTO) launches in the past ten years."_
- The results are exported as a PDF.
- The PDF is then processed using Azure Document Intelligence for RAG (Retrieval-Augmented Generation).

### `/frontend`
- Built with **Streamlit**.
- Communicates with the backend via FastAPI endpoints.

### `/backend`
- Built with **FastAPI** (`main.py`).
- Contains both the API logic and the LLM (Large Language Model) component.
- Machine learning is implemented using a **Linear Regression** model.
- Cross-validation was performed before adapting the model for deployment.

---

## Recommendations to ensure a high-quality solution
### Price Prediction
- I acknowledge that my price prediction model could be improved. When performing cross-validation on the linear regression model, the evaluation metrics (R² and RMSE) were quite poor.  
- This could be due to several factors such as improper encoding of categorical features, using too many or irrelevant features, or simply that a linear regression model may not be well-suited for this problem.  
- Furthermore, since the HDB data is time series in nature—meaning the data points are dependent on each other—time series models like ARIMA or GARCH might be more appropriate for this task, as they can better capture temporal trends and volatility. Given more time, I would explore these approaches to improve prediction accuracy.
### AI / LLM Responses
- I felt that the quality of my AI-generated responses could be improved. To enhance them, I conducted a simple data analysis on one of the prompts—“Please recommend housing estates that have had limited Build-To-Order (BTO) launches in the past ten years”—and stored the analysis results in a PDF, which was then used in the RAG pipeline to supplement the model’s output.  
- Given more time, I would perform a deeper analysis using advanced statistical tests and incorporate additional relevant data such as population size, growth rate, and other demographic factors to refine the recommendations.  
- A more thorough analysis would generate highly informative documents for the Retrieval-Augmented Generation (RAG) pipeline, resulting in more accurate and contextually relevant AI responses.  
- Additionally, as the number of documents used by the RAG pipeline increases, techniques such as text-to-image parsing or knowledge graph integration can be implemented to further enhance the system’s capabilities.

---

Thank you once again for reviewing my work and for this opportunity. I look forward to hearing from you.

---
## If you would like to run the project, please refer to the `setup.md` file for instructions.
