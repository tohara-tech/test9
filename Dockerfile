FROM python:3.12-bullseye

WORKDIR /app

RUN pip install pathlib
RUN pip install python-dotenv
RUN pip install langchain==0.3.7
RUN pip install langchain_chroma==0.1.4
RUN pip install langchain_community==0.3.5
RUN pip install langchain_core==0.3.15
RUN pip install langchain_openai==0.2.6
RUN pip install langchain_text_splitters==0.3.2
RUN pip install streamlit==1.40.0

WORKDIR /workspace/test7/temp
RUN wget https://www.sqlite.org/2024/sqlite-autoconf-3470000.tar.gz
RUN tar xvfz sqlite-autoconf-3470000.tar.gz
WORKDIR /workspace/test7/temp/sqlite-autoconf-3470000
RUN ./configure 
RUN make 
RUN make install
RUN ldconfig

EXPOSE 80
COPY . /app

ENTRYPOINT ["streamlit","run","/app/chatbot.py","--server.port=80"]
