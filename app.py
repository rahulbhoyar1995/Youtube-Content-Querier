import streamlit as st
from llama_index.core import VectorStoreIndex
from llama_index import ServiceContext
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import HuggingFaceInferenceAPI
from llama_hub.youtube_transcript import YoutubeTranscriptReader

# Streamlit title and description
st.title("Youtube Video Content Querier by Rahul Bhoyar")
st.write("Base Model : **HuggingFaceH4/zephyr-7b-alpha (open-source from HuggineFace)**")
st.write("Embedding Model : **WhereIsAI/UAE-Large-V1(open-source from HuggineFace)**")
st.write("This app allows you to read the Youtube video content and tal with it.")

# Streamlit input for user file upload
youtube_url = st.text_input("Enter the youtube video url:")

# Load data and configure the index

    # input_file = save_uploadedfile(uploaded_pdf)
    # st.write("File uploaded successfully!")
    # documents = SimpleDirectoryReader("data").load_data()
if youtube_url:#is not None: 
#if youtube_url is not None: 
    loader = YoutubeTranscriptReader()
    documents = loader.load_data(ytlinks=[youtube_url])
    st.success("Documents loaded successfully!")
    st.markdown("**Here is the content of the Youtube Video:**")
    st.markdown("--" * 15)
    st.markdown(dict(documents[0])['text'])
    st.markdown("--" * 12)


    with st.spinner('Creating Vector Embeddings...'):
        llm = HuggingFaceInferenceAPI(model_name="HuggingFaceH4/zephyr-7b-alpha")
        embed_model_uae = HuggingFaceEmbedding(model_name="WhereIsAI/UAE-Large-V1")


        service_context = ServiceContext.from_defaults(
            llm=llm, chunk_size=800, chunk_overlap=20, embed_model=embed_model_uae
        )
        index = VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True)
        index.storage_context.persist()
        query_engine = index.as_query_engine()
        # Display the result of the task
    st.success("Vector embeddings created.")

    # Streamlit input for user query
    user_query = st.text_input("Enter your query:")

    # Query engine with user input
    if user_query:
        with st.spinner('Fetching the response...'):
            response = query_engine.query(user_query)
            
        st.markdown(f"**Response:** {response}")