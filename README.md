# Deep Learning Project (P12-A)

This README provides a concise guide to understanding and reproducing the steps involved in the deep learning project focused on fine-tuning the `medllama2` model for performing differential diagnosis, utilizing the DDXPlus dataset. Below are the outlined steps and file structures:

## A. Dataset folder

1. **Data Description**: The DDXPlus dataset is stored in the `ddxplus-dataset` folder inside the parent `Dataset` directory. It consists of the following files:

   - `release_condition.json`
   - `release_evidences.json`
   - `test.csv`

2. **Conversion to Human-Readable Text**: The dataset was converted into a more human-readable format stored in `cases.json` using `ddx_data_to_cases.py`. This 'cases.json' file consists of 20k samples taken from the `test.csv` file of DDXPlus.

3. **Conversational Data Generation**: 10k samples from the `cases.json` data were further transformed into conversational form between a human and an assistant acting as a doctor using `generate.py`. The resulting conversational data was stored in `generated_data.csv` and subsequently pushed to the Hugging Face repository [`satyam-03/ddx-conversations-10k`](https://huggingface.co/datasets/satyam-03/ddx-conversations-10k).

## B. Fine-tuning folder

1. **Fine-Tuning Script**: The fine-tuning process was executed using `opdx-alpha.py`. This script is responsible for fine-tuning the `medllama2` model over our generated DDX conversational dataset, resulting in the output stored in the `Model` folder.

## C. Model folder

1. **Model Conversion**: The `convert-lora-to-ggml.py` script in the `llama.cpp` repository, along with `adapter_config.json` and `adapter_model.bin`, were utilized to convert the fine-tuned model into the `ggml` format and generate the model which can be run in inference mode.

For detailed reference on the above tasks, you can refer to [this Medium article](https://sarinsuriyakoon.medium.com/unsloth-lora-with-ollama-lightweight-solution-to-full-cycle-llm-development-edadb6d9e0f0).

## Running the Model

To run the fine-tuned model, follow these steps:

1. **Install Ollama**: Ensure that Ollama is installed on your device.
   ```
   curl -fsSL https://ollama.com/install.sh | sh
   ```
2. **Start the Ollama server**: Run the following command in your terminal:
   ```
   ollama serve
   ```
3. **Build the model from fine-tuned adapter**: In a new terminal, navigate to the Model directory and run the following commands:
   ```
   ollama create opdx -f ./Modelfile
   ```
   Wait for the process to finish, then run:
   ```
   ollama run opdx
   ```

This command will initiate the model, allowing you to perform differential diagnosis using our fine-tuned `opdx` model in the terminal itself.

## Running the UI

### In the UI folder:

1. app folder is the backend of the project.
2. stlit folder contains the frontend made using Streamlit.

### To run the and use the UI of project run the run.sh bash script

### Note: if run.sh didn't run then:

1. Install the requirements.txt in UI folder
2. Navigate to the UI/app directory and run `uvicorn server:app` to start the server.
3. In a new terminal, navigate to the UI/stlit/pages directory and then run `streamlit run ./Home.py`

## Final Project Presentation

You can view the final project presentation [here](https://docs.google.com/presentation/d/11yqWxgCaFN6skNEO1nGk0mxf0vsddR6q1PD7L4g6bXk/edit?usp=sharing).
