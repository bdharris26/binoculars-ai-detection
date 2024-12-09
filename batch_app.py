import gradio as gr
import pandas as pd
import os
from binoculars import Binoculars
import tempfile
from tqdm import tqdm
import plotly.express as px
import random
import re

# Initialize once, not per file
bino = Binoculars()

def clean_text(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = text.replace('\t', ' ')
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = text.strip()
    return text

def generate_variants_from_text(text, method, random_count=50, rolling_words=400):
    """
    Generates a DataFrame with a 'text' column containing various truncated versions
    of the input text according to the chosen method.
    """
    cleaned_text = clean_text(text)
    paragraphs = cleaned_text.split('\n')

    # Start DataFrame with the full cleaned text
    variants = [cleaned_text]

    if method == 'beginning':
        for i in range(1, len(paragraphs)):
            truncated_text = "\n".join(paragraphs[i:])
            variants.append(truncated_text)
    elif method == 'end':
        for i in range(1, len(paragraphs)):
            truncated_text = "\n".join(paragraphs[:-i])
            variants.append(truncated_text)
    elif method == 'random':
        # Generate random_count random variants
        total_pars = len(paragraphs)
        for _ in range(random_count):
            if total_pars == 0:
                break
            start_idx = random.randint(0, total_pars - 1)
            end_idx = random.randint(start_idx, total_pars - 1)
            truncated_pars = paragraphs[start_idx:end_idx+1]
            truncated_text = "\n".join(truncated_pars)
            variants.append(truncated_text)
    elif method == 'rolling':
        WORD_THRESHOLD = rolling_words
        total_pars = len(paragraphs)
        for start_i in range(total_pars):
            word_count = 0
            chunk = []
            for j in range(start_i, total_pars):
                p_words = paragraphs[j].split()
                word_count += len(p_words)
                chunk.append(paragraphs[j])
                if word_count >= WORD_THRESHOLD:
                    truncated_text = "\n".join(chunk)
                    variants.append(truncated_text)
                    break
            else:
                # Can't reach threshold anymore
                break

    return pd.DataFrame({"text": variants})

def create_score_plot(df):
    if df is None or len(df) == 0:
        return None
    fig = px.histogram(
        df, 
        x="raw_score",
        title="Distribution of AI Detection Scores",
        labels={"raw_score": "AI Detection Score"},
        nbins=20
    )
    fig.update_layout(showlegend=False)
    return fig

def run_bino_on_df(df, batch_size):
    """Given a DataFrame with a 'text' column, run bino prediction in batches."""
    if 'text' not in df.columns:
        raise gr.Error("Input DataFrame must contain a 'text' column")

    # Simulate the batching approach
    chunks = []
    for i in tqdm(range(0, len(df), batch_size), desc="Processing Batches"):
        chunk = df.iloc[i:i+batch_size].copy()
        text_list = chunk['text'].tolist()
        predictions = bino.predict(text_list)
        scores = bino.compute_score(text_list)
        chunk['prediction'] = predictions
        chunk['raw_score'] = scores
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    return df

def process_file(file, batch_size, method, random_count, rolling_words):
    if file is None:
        return None, None

    file_ext = os.path.splitext(file.name)[1].lower()

    # If it's a text file, we generate variants first, then run bino
    if file_ext == '.txt':
        try:
            with open(file.name, 'r', encoding='utf-8', errors='replace') as f:
                original_text = f.read()
            df_variants = generate_variants_from_text(original_text, method, random_count, rolling_words)
            df = run_bino_on_df(df_variants, batch_size)
            return df, create_score_plot(df)
        except Exception as e:
            raise gr.Error(f"Error processing text file: {e}")
    # Otherwise, use original CSV/XLSX logic
    elif file_ext == '.csv':
        df_iterator = pd.read_csv(file.name, chunksize=batch_size)
    elif file_ext == '.xlsx':
        df_full = pd.read_excel(file.name)
        df_iterator = (df_full[i:i+batch_size] for i in range(0, len(df_full), batch_size))
    else:
        raise gr.Error("Unsupported file type. Please upload a TXT, CSV or XLSX file.")

    # Original logic for CSV/XLSX
    chunks = []
    for chunk in tqdm(df_iterator, desc="Processing File"):
        if 'text' not in chunk.columns:
            raise gr.Error("Input file must contain a 'text' column")
        
        text_list = chunk['text'].tolist()
        predictions = bino.predict(text_list)
        scores = bino.compute_score(text_list)

        chunk['prediction'] = predictions
        chunk['raw_score'] = scores
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    return df, create_score_plot(df)


def save_df(df):
    if df is None:
        return None
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "results.xlsx")
    try:
        df.to_excel(output_path, index=False)
        return output_path
    except Exception as e:
        raise gr.Error(f"Error saving results: {str(e)}")

def batch_interface():
    with gr.Blocks(css="""
        .wrap-text {
            max-width: 400px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .plot-container {
            height: 300px;
        }
        .fixed-height-row {
            max-height: 1000px !important;
            overflow-y: auto !important;
        }
        .fixed-height-row td {
            max-height: 100px !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }
        """) as demo:

        gr.Markdown("## Batch Processing for AI Text Detection")
        gr.Markdown("You can upload a .txt file to generate variants or a CSV/XLSX with a 'text' column directly.")
        
        with gr.Row():
            file_input = gr.File(
                label="Upload TXT, CSV, or XLSX",
                file_types=[".txt", ".csv", ".xlsx"],
                type="filepath"
            )
            batch_size = gr.Slider(
                minimum=1,
                maximum=32,
                value=8,
                step=1,
                label="Batch Size"
            )

        # Method selection
        with gr.Row():
            method = gr.Radio(
                choices=["beginning", "end", "random", "rolling"],
                value="beginning",
                label="Text Generation Method (For TXT Files)"
            )

        with gr.Row():
            random_count = gr.Number(value=50, label="Number of random rows (For 'random' method)")
            rolling_words = gr.Number(value=400, label="Word count threshold (For 'rolling' method)")

        with gr.Row():
            output = gr.Dataframe(
                headers=["text", "prediction", "raw_score"],
                datatype=["str", "str", "number"],
                visible=True,
                wrap=True,
                column_widths=["65%", "20%", "15%"],
                elem_classes=["fixed-height-row"]
            )

        with gr.Row():
            plot = gr.Plot(label="Score Distribution")

        with gr.Row():
            download_button = gr.Button("Download Results")

        file_input.change(
            fn=process_file,
            inputs=[file_input, batch_size, method, random_count, rolling_words],
            outputs=[output, plot]
        )

        output.change(
            fn=create_score_plot,
            inputs=[output],
            outputs=[plot]
        )
        
        download_button.click(
            fn=save_df,
            inputs=output,
            outputs=gr.File(label="Download Results", file_types=[".xlsx"])
        )
    
    return demo

if __name__ == "__main__":
    batch_interface().launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )
