import gradio as gr
import pandas as pd
import os
import argparse
from binoculars import Binoculars
import tempfile
from tqdm import tqdm
import plotly.express as px
import random
import re
import datetime
import math
import statistics

OPTIMAL_CHAR_LENGTH = 1242  # Approximate optimal length for chunking

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

def split_text_into_chunks(text, max_length=OPTIMAL_CHAR_LENGTH):
    """
    Splits text into chunks by line boundaries, aiming for each chunk to be close to but not exceed max_length.
    """
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0

    for line in lines:
        line_length = len(line) + 1  # +1 for the newline
        if current_length + line_length > max_length and current_chunk:
            # Close off the current chunk and start a new one
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_length = line_length
        else:
            current_chunk.append(line)
            current_length += line_length

    # Add the last chunk if present
    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks if chunks else [text]

def generate_variants_from_text(text, method, random_count=50, rolling_words=400):
    cleaned_text = clean_text(text)
    if not cleaned_text:
        # If no text, return empty DataFrame
        return pd.DataFrame({"text": []})
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
    if 'raw_score' not in df.columns:
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

def run_bino_on_df(df, batch_size, bino):
    if 'text' not in df.columns:
        raise gr.Error("Input DataFrame must contain a 'text' column")

    # If DF is empty, notify user
    if df.empty:
        raise gr.Error("No text variants or chunks were generated. The input might be empty or invalid.")

    chunks = []
    for i in tqdm(range(0, len(df), batch_size), desc="Processing Batches"):
        chunk = df.iloc[i:i+batch_size].copy()
        text_list = chunk['text'].tolist()
        try:
            predictions = bino.predict(text_list)
            scores = bino.compute_score(text_list)
        except Exception as e:
            raise gr.Error(f"Error during model inference: {e}")

        chunk['prediction'] = predictions
        chunk['raw_score'] = scores
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    return df

def compute_statistics(scores):
    mean_val = statistics.mean(scores) if scores else 0
    median_val = statistics.median(scores) if scores else 0
    stdev_val = statistics.pstdev(scores) if len(scores) > 1 else 0
    min_val = min(scores) if scores else 0
    max_val = max(scores) if scores else 0
    n = len(scores)
    # Compute 95% CI if n > 1
    ci_lower = ci_upper = mean_val
    if n > 1:
        ci_margin = 1.96 * (stdev_val / math.sqrt(n))
        ci_lower = mean_val - ci_margin
        ci_upper = mean_val + ci_margin

    return {
        "mean_score": mean_val,
        "median_score": median_val,
        "std_dev": stdev_val,
        "min_score": min_val,
        "max_score": max_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "chunk_count": n
    }

def process_txt_file(file, batch_size, method, random_count, rolling_words, bino):
    with open(file.name, 'r', encoding='utf-8', errors='replace') as f:
        original_text = f.read()
    df_variants = generate_variants_from_text(original_text, method, random_count, rolling_words)
    if df_variants.empty:
        raise gr.Error("No variants generated from the provided text file. The file may be empty.")

    # Apply chunking to each variant
    chunked_rows = []
    for idx, row in df_variants.iterrows():
        text = row['text']
        chunks = split_text_into_chunks(text, OPTIMAL_CHAR_LENGTH)
        for ch in chunks:
            chunked_rows.append({"original_index": idx, "text": ch})

    chunked_df = pd.DataFrame(chunked_rows)
    df = run_bino_on_df(chunked_df, batch_size, bino)

    # Aggregate stats per original variant
    summary_rows = []
    for idx, group in df.groupby("original_index"):
        scores = group['raw_score'].tolist()
        stats = compute_statistics(scores)
        summary_row = {
            "original_index": idx,
            "mean_score": stats["mean_score"],
            "median_score": stats["median_score"],
            "std_dev": stats["std_dev"],
            "min_score": stats["min_score"],
            "max_score": stats["max_score"],
            "ci_lower": stats["ci_lower"],
            "ci_upper": stats["ci_upper"],
            "chunk_count": stats["chunk_count"]
        }
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    return df, summary_df

def process_tabular_file(file, batch_size, bino):
    # Works for both XLSX and CSV by reading into a single dataframe
    file_ext = os.path.splitext(file.name)[1].lower()
    if file_ext == '.csv':
        df_full = pd.read_csv(file.name)
    else:
        df_full = pd.read_excel(file.name)

    if 'text' not in df_full.columns:
        raise gr.Error("Input file must contain a 'text' column")

    # Apply chunking per row if text is too long
    expanded_rows = []
    for i, row in df_full.iterrows():
        text = clean_text(row['text'])
        chunks = split_text_into_chunks(text, OPTIMAL_CHAR_LENGTH)
        for ch in chunks:
            new_row = row.copy()
            new_row['text'] = ch
            new_row['original_id'] = i
            expanded_rows.append(new_row)

    expanded_df = pd.DataFrame(expanded_rows)
    processed_df = run_bino_on_df(expanded_df, batch_size, bino)

    # Aggregate stats by original_id
    summary_rows = []
    for i, group in processed_df.groupby('original_id'):
        scores = group['raw_score'].tolist()
        stats = compute_statistics(scores)
        summary_row = {
            "original_id": i,
            "mean_score": stats["mean_score"],
            "median_score": stats["median_score"],
            "std_dev": stats["std_dev"],
            "min_score": stats["min_score"],
            "max_score": stats["max_score"],
            "ci_lower": stats["ci_lower"],
            "ci_upper": stats["ci_upper"],
            "chunk_count": stats["chunk_count"]
        }
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    return processed_df, summary_df

def process_file(file, batch_size, method, random_count, rolling_words, bino):
    if file is None:
        return None, None, None

    file_ext = os.path.splitext(file.name)[1].lower()

    if file_ext == '.txt':
        try:
            detail_df, summary_df = process_txt_file(file, batch_size, method, random_count, rolling_words, bino)
            plot = create_score_plot(detail_df)
            return detail_df, plot, summary_df
        except Exception as e:
            raise gr.Error(f"Error processing text file: {e}")

    elif file_ext in ['.csv', '.xlsx']:
        try:
            detail_df, summary_df = process_tabular_file(file, batch_size, bino)
            plot = create_score_plot(detail_df)
            return detail_df, plot, summary_df
        except Exception as e:
            raise gr.Error(f"Error processing file: {e}")

    else:
        raise gr.Error("Unsupported file type. Please upload a TXT, CSV or XLSX file.")

def save_df(df):
    if df is None or df.empty:
        return None
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"results_{timestamp}.xlsx"
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, output_filename)
    try:
        df.to_excel(output_path, index=False)
        return output_path
    except Exception as e:
        raise gr.Error(f"Error saving results: {str(e)}")

def batch_interface(bino):
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

        gr.Markdown("## Batch Processing for AI Text Detection with Chunking and Statistics")
        gr.Markdown("Upload a .txt file to generate variants and chunks, or a CSV/XLSX to process multiple texts. Results will include statistical insights. Two tables are shown: one for detailed chunk-level results and another for aggregated summaries.")

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
                label="Variant Generation Method (For TXT Files)"
            )

        with gr.Row():
            random_count = gr.Number(value=50, label="Number of random rows (For 'random' method)")
            rolling_words = gr.Number(value=400, label="Word count threshold (For 'rolling' method)")

        gr.Markdown("### Detailed Results (Chunk-level)")
        detail_output = gr.Dataframe(
            headers=["text", "prediction", "raw_score", "mean_score", "median_score", "std_dev", "min_score", "max_score", "ci_lower", "ci_upper", "chunk_count"],
            datatype=["str", "str", "number", "number", "number", "number", "number", "number", "number", "number", "number"],
            visible=True,
            wrap=True,
            column_widths=["30%", "10%", "10%", "5%", "5%", "5%", "5%", "5%", "5%", "5%", "5%"],
            elem_classes=["fixed-height-row"]
        )

        gr.Markdown("### Summary Results (Per Original Text)")
        summary_output = gr.Dataframe(
            headers=["original_index/original_id", "mean_score", "median_score", "std_dev", "min_score", "max_score", "ci_lower", "ci_upper", "chunk_count"],
            datatype=["number", "number", "number", "number", "number", "number", "number", "number", "number"],
            visible=True,
            wrap=True,
            column_widths=["10%", "10%", "10%", "10%", "10%", "10%", "10%", "10%", "10%"],
            elem_classes=["fixed-height-row"]
        )

        with gr.Row():
            plot = gr.Plot(label="Score Distribution")

        with gr.Row():
            download_button = gr.Button("Download Detailed Results")
            download_summary_button = gr.Button("Download Summary Results")

        file_input.change(
            fn=lambda file, bs, m, rc, rw: process_file(file, bs, m, rc, rw, bino),
            inputs=[file_input, batch_size, method, random_count, rolling_words],
            outputs=[detail_output, plot, summary_output]
        )

        detail_output.change(
            fn=create_score_plot,
            inputs=[detail_output],
            outputs=[plot]
        )

        download_button.click(
            fn=save_df,
            inputs=detail_output,
            outputs=gr.File(label="Download Detailed Results", file_types=[".xlsx"])
        )

        download_summary_button.click(
            fn=save_df,
            inputs=summary_output,
            outputs=gr.File(label="Download Summary Results", file_types=[".xlsx"])
        )

    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--observer", default="tiiuae/falcon-7b", help="Observer model name or path")
    parser.add_argument("--performer", default="tiiuae/falcon-7b-instruct", help="Performer model name or path")
    args = parser.parse_args()

    try:
        bino = Binoculars(
            observer_name_or_path=args.observer,
            performer_name_or_path=args.performer
        )
    except Exception as e:
        raise SystemExit(f"Failed to initialize Binoculars: {e}")

    batch_interface(bino).launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )
