# Run FastAPI server using uvicorn in the background
cd UI/app
uvicorn server:app &

# Run Streamlit app
cd ../stlit/pages
streamlit run Home.py