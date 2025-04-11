🌟 Application Name : FINACLE 🌟 

Team Name: Lancelot Team members: Sree Snehan - Kuhan M - Anto Nickson J

Mail: snehandot@gmail.com   Contact No: 98949 37066 , 78450 36121

🚀 Embark on a Financial Journey with FINACLE!📈

Welcome to our revolutionary tool designed to empower your financial decisions effortlessly.

!Introduction 💻

Finacle is an innovative application meticulously crafted to streamline your financial analysis process. Whether you're a seasoned investor or just starting your journey, our platform offers unparalleled insights into mutual fund schemes. Say goodbye to tedious manual analysis of PDF reports — Finacle automates the process, extracting scheme names and their corresponding category details for the month.

How to use ! ⬆

🔑 Input OpenAI API Key: Upload the API key sent personally through Whatsapp or feel free to use your own.Contact us for any issues or query related to API key:)

📄 Upload PDF Document: Upload the pdf file through the Browse button , the sample pdfs come pre-loaded into our application.

🔄 Process PDF: Once the document is uploaded, click the "Process" button to store your pdf into the vector database. Ensure that the same file isn't processed again to avoid duplication.

🔍 Select Scheme Details: Select your required scheme details in the fields.

🏃‍♂ Run Model: Click "Submit" , after selecting the fields and wait for a few seconds to get our analysis. Wait till the graph is displayed.

📥 Download CSV: Once the graph is displayed download the CSV if required.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

⚠ Alert: Users are advised to download the CSV file after viewing the graphs. Clicking the download csv button may reload the application.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Key Features ! 🚀

-Automated Analysis: Effortlessly analyze PDF reports containing scheme names and their diverse category details for the month. 📊We have

-Comprehensive Insights: Gain comprehensive insights into mutual fund schemes, enabling informed decision-making and -portfolio optimization. 📈

-User-Friendly Interface: Navigate our intuitive platform with ease, allowing for seamless interaction and efficient -utilization of data. 💻

-Time-Saving Solution: Save valuable time by automating the analysis process, freeing up resources for strategic investment planning and execution. ⏱

Technical Aspects:

-> PyPdf is used to retreive the data from the PDF.

-> The chunks are split and directed to the embedding model "text-embedding-3-small" from OPEN AI to store into the vector database

-> The vectorised data is embedded into the local FAISS vector database.

-> The LLM model "GPT-4-Turbo-2024-04-09" is connected to the FAISS database for similarity and semantic search for quick and efficient process.

-> For creating a retreival QA_chain we have used load_q_a_chain from LangChain.chains.

-> For prompt-specifying we have used Prompt template from Langchain.prompts.

Update!

2nd Place
![2nd Place](https://github.com/user-attachments/assets/b03f4fa8-d408-46d4-82fc-20e7a4386e6a)
