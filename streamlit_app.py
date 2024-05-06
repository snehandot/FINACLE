import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores.faiss import FAISS
# from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import FewShotPromptTemplate
import pandas as pd
import re
import matplotlib.pyplot as plt
import openai
import requests
import json, ast, csv

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text




def get_Chat_response(chain, docs, user_query):
    a = chain.run(input_documents=docs, question=user_query)
    output = (a[a.index('Helpful Answer:')+16:])
    return output


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=10000,
        chunk_overlap=100   ,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")



def get_conversational_chain():

    prompt_template = """
    Question: Open ended Schemes Income/Debt Oriented Schemes  Overnight Fund November 2023

    Answer : No. of Schemes as on November 30, 2023 : 35 
             No. of Folios as on November 30, 2023 : 8,29,534
             Funds Mobilized for the month of November 2023 (INR in crore) : 4,87,463.14
             Repurchase/ Redemption for the month of November 2023 (INR in crore) : 4,87,947.50
             Net Assets Under Management as on November 30, 2023 (INR in crore) : 79,291.30 
             Average Net Assets Under Management for the month November 2023 (INR in crore) :  1,05,689.07
             No. of segregated portfolios created as on November 30, 2023 : -
             Net Assets Under Management in segregated portfolio as on November 30, 2023 (INR in crore) : -

             You Want to calculate the Following Net Inflow or Outflow,Net Asset under Management per Scheme, Net Inflow or Outflow per Scheme the formula were given below 
            - Net Inflow or Outflow: Calculated as Funds Mobilized for the month - Repurchase/Redemption for the month.
            - Net Asset under Management per Scheme: Determined by dividing the Net Assets Under Management by the No. of Schemes.
            - Net Inflow or Outflow per Scheme: This is the ratio of Net Inflow or Outflow to the No. of Schemes.


    Question: Open ended Schemes Income/Debt Oriented Schemes  Overnight Fund November 2023

    Answer : No. of Schemes as on November 30, 2023 : 35 
             No. of Folios as on November 30, 2023 : 8,29,534
             Funds Mobilized for the month of November 2023 (INR in crore) : 4,87,463.14
             Repurchase/ Redemption for the month of November 2023 (INR in crore) : 4,87,947.50
             Net Assets Under Management as on November 30, 2023 (INR in crore) : 79,291.30 
             Average Net Assets Under Management for the month November 2023 (INR in crore) :  1,05,689.07
             No. of segregated portfolios created as on November 30, 2023 : -
             Net Assets Under Management in segregated portfolio as on November 30, 2023 (INR in crore) : -

             You Want to calculate the Following Net Inflow or Outflow,Net Asset under Management per Scheme, Net Inflow or Outflow per Scheme the formula were given below 
            - Net Inflow or Outflow: Calculated as Funds Mobilized for the month - Repurchase/Redemption for the month.
            - Net Asset under Management per Scheme: Determined by dividing the Net Assets Under Management by the No. of Schemes.
            - Net Inflow or Outflow per Scheme: This is the ratio of Net Inflow or Outflow to the No. of Schemes.



    Question: Open ended Schemes Growth/Equity Oriented Schemes Multi Cap Fund November 2023

    Answer : No. of Schemes as on November 30, 2023 : 23 
             No. of Folios as on November 30, 2023 : 51,43,687
             Funds Mobilized for the month of November 2023 (INR in crore) : 2,781.78
             Repurchase/ Redemption for the month of November 2023 (INR in crore) : 1,068.70
             Net Assets Under Management as on November 30, 2023 (INR in crore) : 1,03,056.62
             Average Net Assets Under Management for the month November 2023 (INR in crore) :  98,905.53
             No. of segregated portfolios created as on November 30, 2023 : -
             Net Assets Under Management in segregated portfolio as on November 30, 2023 (INR in crore) : -

             You Want to calculate the Following Net Inflow or Outflow,Net Asset under Management per Scheme, Net Inflow or Outflow per Scheme the formula were given below 
            - Net Inflow or Outflow: Calculated as Funds Mobilized for the month - Repurchase/Redemption for the month.
            - Net Asset under Management per Scheme: Determined by dividing the Net Assets Under Management by the No. of Schemes.
            - Net Inflow or Outflow per Scheme: This is the ratio of Net Inflow or Outflow to the No. of Schemes.


Question: Open ended Schemes Hybrid Schemes Conservative Hybrid Fund November 2023

    Answer : No. of Schemes as on November 30, 2023 : 20 
             No. of Folios as on November 30, 2023 : 5,32,921
             Funds Mobilized for the month of November 2023 (INR in crore) : 367.82
             Repurchase/ Redemption for the month of November 2023 (INR in crore) : 364.95
             Net Assets Under Management as on November 30, 2023 (INR in crore) : 25,729.94 
             Average Net Assets Under Management for the month November 2023 (INR in crore) :  25,535.91
             No. of segregated portfolios created as on November 30, 2023 : 1
             Net Assets Under Management in segregated portfolio as on November 30, 2023 (INR in crore) : -

             You Want to calculate the Following Net Inflow or Outflow,Net Asset under Management per Scheme, Net Inflow or Outflow per Scheme the formula were given below 
            - Net Inflow or Outflow: Calculated as Funds Mobilized for the month - Repurchase/Redemption for the month.
            - Net Asset under Management per Scheme: Determined by dividing the Net Assets Under Management by the No. of Schemes.
            - Net Inflow or Outflow per Scheme: This is the ratio of Net Inflow or Outflow to the No. of Schemes.


Question: Open ended Schemes Solution Oriented Schemes Retirement Fund November 2023

    Answer : No. of Schemes as on November 30, 2023 : 27 
             No. of Folios as on November 30, 2023 : 28,30,927
             Funds Mobilized for the month of November 2023 (INR in crore) : 226.40
             Repurchase/ Redemption for the month of November 2023 (INR in crore) : 118.92
             Net Assets Under Management as on November 30, 2023 (INR in crore) : 22,680.43 
             Average Net Assets Under Management for the month November 2023 (INR in crore) :  22,102.36
             No. of segregated portfolios created as on November 30, 2023 : -
             Net Assets Under Management in segregated portfolio as on November 30, 2023 (INR in crore) : -

             You Want to calculate the Following Net Inflow or Outflow,Net Asset under Management per Scheme, Net Inflow or Outflow per Scheme the formula were given below 
            - Net Inflow or Outflow: Calculated as Funds Mobilized for the month - Repurchase/Redemption for the month.
            - Net Asset under Management per Scheme: Determined by dividing the Net Assets Under Management by the No. of Schemes.
            - Net Inflow or Outflow per Scheme: This is the ratio of Net Inflow or Outflow to the No. of Schemes.


Question: Open ended Schemes Other Schemes Index Funds November 2023

    Answer : No. of Schemes as on November 30, 2023 : 193 
             No. of Folios as on November 30, 2023 : 59,69,122
             Funds Mobilized for the month of November 2023 (INR in crore) : 4,081.61
             Repurchase/ Redemption for the month of November 2023 (INR in crore) : 2,728.15
             Net Assets Under Management as on November 30, 2023 (INR in crore) : 1,91,412.86 
             Average Net Assets Under Management for the month November 2023 (INR in crore) : 1,88,343.82
             No. of segregated portfolios created as on November 30, 2023 : -
             Net Assets Under Management in segregated portfolio as on November 30, 2023 (INR in crore) : -

             You Want to calculate the Following Net Inflow or Outflow,Net Asset under Management per Scheme, Net Inflow or Outflow per Scheme the formula were given below 
            - Net Inflow or Outflow: Calculated as Funds Mobilized for the month - Repurchase/Redemption for the month.
            - Net Asset under Management per Scheme: Determined by dividing the Net Assets Under Management by the No. of Schemes.
            - Net Inflow or Outflow per Scheme: This is the ratio of Net Inflow or Outflow to the No. of Schemes.


Question: Close Ended Schemes Income/Debt Oriented Schemes Fixed Term Plan November 2023

    Answer : No. of Schemes as on November 30, 2023 : 84 
             No. of Folios as on November 30, 2023 : 78,979
             Funds Mobilized for the month of November 2023 (INR in crore) : 446.62
             Repurchase/ Redemption for the month of November 2023 (INR in crore) : 1,675.42
             Net Assets Under Management as on November 30, 2023 (INR in crore) : 15,656.98 
             Average Net Assets Under Management for the month November 2023 (INR in crore) :  15,840.25
             No. of segregated portfolios created as on November 30, 2023 : -
             Net Assets Under Management in segregated portfolio as on November 30, 2023 (INR in crore) : -

             You Want to calculate the Following Net Inflow or Outflow,Net Asset under Management per Scheme, Net Inflow or Outflow per Scheme the formula were given below 
            - Net Inflow or Outflow: Calculated as Funds Mobilized for the month - Repurchase/Redemption for the month.
            - Net Asset under Management per Scheme: Determined by dividing the Net Assets Under Management by the No. of Schemes.
            - Net Inflow or Outflow per Scheme: This is the ratio of Net Inflow or Outflow to the No. of Schemes.


Question: Close Ended Schemes Growth/Equity Oriented Schemes ELSS November 2023

    Answer : No. of Schemes as on November 30, 2023 : 19 
             No. of Folios as on November 30, 2023 : 2,84,776
             Funds Mobilized for the month of November 2023 (INR in crore) : - 
             Repurchase/ Redemption for the month of November 2023 (INR in crore) : 21.93
             Net Assets Under Management as on November 30, 2023 (INR in crore) : 3,893.79 
             Average Net Assets Under Management for the month November 2023 (INR in crore) :  3,797.43
             No. of segregated portfolios created as on November 30, 2023 : -
             Net Assets Under Management in segregated portfolio as on November 30, 2023 (INR in crore) : -

             You Want to calculate the Following Net Inflow or Outflow,Net Asset under Management per Scheme, Net Inflow or Outflow per Scheme the formula were given below 
            - Net Inflow or Outflow: Calculated as Funds Mobilized for the month - Repurchase/Redemption for the month.
            - Net Asset under Management per Scheme: Determined by dividing the Net Assets Under Management by the No. of Schemes.
            - Net Inflow or Outflow per Scheme: This is the ratio of Net Inflow or Outflow to the No. of Schemes.



Question: Close Ended Schemes Growth/Equity Oriented Schemes ELSS November 2023

    Answer : No. of Schemes as on November 30, 2023 : 19 
             No. of Folios as on November 30, 2023 : 2,84,776
             Funds Mobilized for the month of November 2023 (INR in crore) : - 
             Repurchase/ Redemption for the month of November 2023 (INR in crore) : 21.93
             Net Assets Under Management as on November 30, 2023 (INR in crore) : 3,893.79 
             Average Net Assets Under Management for the month November 2023 (INR in crore) :  3,797.43
             No. of segregated portfolios created as on November 30, 2023 : -
             Net Assets Under Management in segregated portfolio as on November 30, 2023 (INR in crore) : -

             You Want to calculate the Following Net Inflow or Outflow,Net Asset under Management per Scheme, Net Inflow or Outflow per Scheme the formula were given below 
            - Net Inflow or Outflow: Calculated as Funds Mobilized for the month - Repurchase/Redemption for the month.
            - Net Asset under Management per Scheme: Determined by dividing the Net Assets Under Management by the No. of Schemes.
            - Net Inflow or Outflow per Scheme: This is the ratio of Net Inflow or Outflow to the No. of Schemes.


Question: Interval Schemes Income/Debt Oriented Schemes November 2023

    Answer : No. of Schemes as on November 30, 2023 : 12 
             No. of Folios as on November 30, 2023 : 2,619
             Funds Mobilized for the month of November 2023 (INR in crore) : 134.11 
             Repurchase/ Redemption for the month of November 2023 (INR in crore) : 2.84
             Net Assets Under Management as on November 30, 2023 (INR in crore) : 407.71 
             Average Net Assets Under Management for the month November 2023 (INR in crore) :  334.22
             No. of segregated portfolios created as on November 30, 2023 : -
             Net Assets Under Management in segregated portfolio as on November 30, 2023 (INR in crore) : -

             You Want to calculate the Following Net Inflow or Outflow,Net Asset under Management per Scheme, Net Inflow or Outflow per Scheme the formula were given below 
            - Net Inflow or Outflow: Calculated as Funds Mobilized for the month - Repurchase/Redemption for the month.
            - Net Asset under Management per Scheme: Determined by dividing the Net Assets Under Management by the No. of Schemes.
            - Net Inflow or Outflow per Scheme: This is the ratio of Net Inflow or Outflow to the No. of Schemes.




           
 
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatOpenAI(model='gpt-4-turbo-2024-04-09', temperature=1)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def gpt4(prompt):
    import os
    openai.api_key = os.environ['OPENAI_API_KEY']
    response = openai.ChatCompletion.create(
    model="gpt-4",
    max_tokens=800,
    temperature=1.2,
    messages = [{"role": "user", "content": prompt}])
    
    return response




def handle_userinput(user_question):
    embeddings = OpenAIEmbeddings()
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

  

    
    st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", response['output_text']), unsafe_allow_html=True)



    question = response['output_text']

    response = gpt4('''Sample Input Text: No. of Schemes as on January 31, 2021: 30 No. of Folios as on January 31, 2021: 3,07,669 Funds Mobilized for the month of January 2021 (INR in crore): 4,53,336.11 Repurchase/ Redemption for the month of January 2021 (INR in crore): 4,48,605.13 Net Assets Under Management as on January 31, 2021 (INR in crore): 94,138.69 Average Net Assets Under Management for the month January 2021 (INR in crore): 110,033.81 No. of segregated portfolios created as on January 31, 2021: - Net Assets Under Management in segregated portfolio as on January 31, 2021 (INR in crore): -
        # You want to calculate the following:

        Net Inflow or Outflow: Calculated as Funds Mobilized for the month - Repurchase/Redemption for the month.
        Net Asset under Management per Scheme: Determined by dividing the Net Assets Under Management by the No. of Schemes.
        Net Inflow or Outflow per Scheme: This is the ratio of Net Inflow or Outflow to the No. of Schemes.
        Let's compute these values:

        Net Inflow or Outflow: ( 4,53,336.11 - 4,48,605.13 = 4,730.98 ) crore INR

        Net Asset under Management per Scheme: ( \frac{94,138.69}{30} \approx 3137.96 ) crore INR per scheme

        Net Inflow or Outflow per Scheme: ( \frac{4730.98}{30} \approx 157.70 ) crore INR per scheme

        These calculations detail the financial movements and holdings for the Overnight Fund category within open-ended income/debt oriented schemes as of January 2021.

        I need output in the format of list given below no text should be in the response only a list
        [{"Metric": "No. of Schemes as on January 31, 2021", "Value": 30},{"Metric": "No. of Folios as on January 31, 2021", "Value": 307669},{"Metric": "Funds Mobilized for the month of January 2021 (INR in crore)", "Value": 453336.11},{"Metric": "Repurchase/ Redemption for the month of January 2021 (INR in crore)", "Value": 448605.13},{"Metric": "Net Assets Under Management as on January 31, 2021 (INR in crore)", "Value": 94138.69},{"Metric": "Average Net Assets Under Management for the month January 2021 (INR in crore)", "Value": 110033.81},{"Metric": "No. of segregated portfolios created as on January 31, 2021", "Value": None},{"Metric": "Net Assets Under Management in segregated portfolio as on January 31, 2021 (INR in crore)", "Value": None},{"Metric": "Net Inflow or Outflow", "Value": 4730.98},{"Metric": "Net Asset under Management per Scheme", "Value": 3137.96},{"Metric": "Net Inflow or Outflow per Scheme", "Value": 157.70}]
        and now your question is:
        Question:'''+question+"No additional text only give the list [ ] as output no other notaions in results like output:")
    

    generated_content = response['choices'][0]['message']['content']
    
    data_lis = eval(generated_content)

    output_file = "financia_data.csv"

    # Write the data to the CSV file
    with open(output_file, "w", newline="") as csvfile:
            fieldnames = ["Metric", "Value"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write the header
            writer.writeheader()

            # Write the data rows
            for row in data_lis:
                writer.writerow(row)


    data = pd.read_csv(output_file)

        # Provide download button
    st.write("Dowload the CSV of the Requested Data (Download after analysing the graph and given data)")
    st.download_button(
            label="Download CSV",
            data=data.to_csv().encode('utf-8'),
            file_name='financial_data.csv',
            mime='text/csv',
        )    







  





# Load the data
df = pd.read_csv('combined_file.csv')



# Function to plot the data
def plot_data(string, string_list, df):
    main = None
    for s in string_list:
        if s in string:
            main = s
            break

    if main is None:
        st.error("No matching string found in the list.")
        return

    # Filter the dataframe
    filtered_df = df[df.iloc[:, 1] == main]

    # Extracting values for month, good_values, and bad_values
    month = filtered_df.iloc[:, 0].tolist()
    good_values = [re.sub(r'[^\d.]', '', str(i)) for i in filtered_df.iloc[:, 4].tolist()]
    good_values = [float(i) for i in good_values]
    good_values = good_values[::-1]
    bad_values = [re.sub(r'[^\d.]', '', str(i)) for i in filtered_df.iloc[:, 5].tolist()]
    bad_values = [float(i) for i in bad_values]
    bad_values = bad_values[::-1]

    # Plotting the data
    plt.plot(month, good_values, label='Mobilized')
    plt.plot(month, bad_values, label='Redeemed')

    # Adding labels and title
    plt.xlabel('Month')
    plt.ylabel('Values')
    plt.title(main + " over the years")
    plt.legend()
    st.write("Graph : Previous year Analysis")
    # Displaying the plot
    st.pyplot()



def main():
    load_dotenv()
    st.set_page_config(page_title="FINACLE FUND ANALYSER 🔍")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write(css, unsafe_allow_html=True)


    # CSS style for centering the select boxes
    center_style = """
        <style>
            .center {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
        
        </style>
    """

    # Display CSS style
    st.markdown(center_style, unsafe_allow_html=True)

    # Create a dictionary of categories and their options
    categories = {
        'Scheme Type': [' Open ended Schemes', ' Close Ended Schemes', 'Interval Schemes'],
    'Scheme Name': ['Income/Debt Oriented Schemes', 'Growth/Equity Oriented Schemes', 'Hybrid Schemes','Solution Oriented Schemes','Other Schemes'],
    'Fund Name': [' Overnight Fund', 'Liquid Fund', 'Ultra Short Duration Fund',' Low Duration Fund','Money Market Fund','Short Duration Fund','Medium Duration Fund','Medium to Long Duration Fund','Long Duration Fund','Dynamic Bond Fund','Corporate Bond Fund','Credit Risk Fund',' Banking and PSU Fund','Gilt Fund','Gilt Fund with 10 year constant duration','Floater Fund','Multi Cap Fund','Large Cap Fund','Large & Mid Cap Fund',' Mid Cap Fund',' Small Cap Fund',' Dividend Yield Fund',' Value Fund/Contra Fund',' Focused Fund','Sectoral/Thematic Funds','ELSS','Flexi Cap Fund','Conservative Hybrid Fund','Balanced Hybrid Fund/Aggressive Hybrid Fund','Dynamic Asset Allocation/Balanced Advantage Fund','Multi Asset Allocation Fund','Arbitrage Fund','Equity Savings Fund','Retirement Fund','Childrens Fund','Index Funds','GOLD ETF','Other ETFs','Fund of funds investing overseas','Fixed Term Plan','Capital Protection Oriented Scheme','Infrastructure Debt Fund','Other Debt Scheme','Other Equity Schemes','','',],
    # 'Attributes': ['No. of Schemes', 'No. of Folios', 'Funds Mobilized','Repurchase/Redemption','Net Assets Under Management','Average Net Assets Under Management','No. of segregat ed portfolios created','Net Assets Under Managem ent in segregate d portfolio'],
    'Month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
    'Year': ['2021','2022', '2023','2024']

    }

    # Create variables to store selected options
    selected_options = {category: None for category in categories}

    # Create selectbox widgets for each category
    for category, options in categories.items():
        selected_options[category] = st.selectbox(f'Select {category}', options)
    # Store selected options in backend variables
    category_1_selected = selected_options['Scheme Type']
    category_2_selected = selected_options['Scheme Name']
    category_3_selected = selected_options['Fund Name']
    # category_4_selected = selected_options['Attributes']
    category_5_selected = selected_options['Month']
    category_6_selected = selected_options['Year']
    # Add a submit button
    if st.button('Submit'):
        # Display selected options
        st.write('You selected:')
        for category, option in selected_options.items():
            st.write(f'- {category}: {option}')

        if "conversation" not in st.session_state:
            st.session_state.conversation = None
            st.header("Finacle fund analyser")

        user_question= category_1_selected + " "+ category_2_selected +" " + category_3_selected +" " + category_5_selected +" " + category_6_selected
        if user_question:
            handle_userinput(user_question)


        string = selected_options['Fund Name']

        string_list = ['Overnight Fund', 'Liquid Fund', 'Ultra Short Duration Fund', 'Low Duration Fund',
                   'Money Market Fund', 'Short Duration Fund', 'Medium Duration Fund', 'Medium to Long Duration Fund',
                   'Long Duration Fund', 'Dynamic Bond Fund', 'Corporate Bond Fund', 'Credit Risk Fund',
                   'Banking and PSU Fund', 'Gilt Fund', 'Gilt Fund with 10 year constant duration', 'Floater Fund',
                   'Multi Cap Fund', 'Large Cap Fund', 'Large & Mid Cap Fund', 'Mid Cap Fund', 'Small Cap Fund',
                   'Dividend Yield Fund', 'Value Fund/Contra Fund', 'Focused Fund', 'Sectoral/Thematic Funds', 'ELSS',
                   'Flexi Cap Fund', 'Conservative Hybrid Fund', 'Balanced Hybrid Fund/Aggressive Hybrid Fund',
                   'Dynamic Asset Allocation/Balanced Advantage Fund', 'Multi Asset Allocation Fund', 'Arbitrage Fund',
                   'Equity Savings Fund', 'Retirement Fund', 'Childrens Fund', 'Index Funds', 'GOLD ETF', 'Other ETFs',
                   'Fund of funds investing overseas', 'Fixed Term Plan', 'Capital Protection Oriented Scheme',
                   'Infrastructure Debt Fund', 'Other Debt Scheme', 'Other Equity Schemes']
        plot_data(string, string_list, df)
    




    








    with st.sidebar:

        import os
        api_key = st.text_input('OpenAI API Key:', type='password')
        openai.api_key = api_key
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                




if __name__ == '__main__':
    main()
