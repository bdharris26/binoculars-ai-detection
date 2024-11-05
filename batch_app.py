import gradio as gr
import pandas as pd
from binoculars import Binoculars

# Initialize the Binoculars model
bino = Binoculars()

def process_csv(file):
    """
    Process the uploaded CSV file and return predictions and raw scores.

    Args:
        file (file): The uploaded CSV file.

    Returns:
        pd.DataFrame: The DataFrame containing the text, predictions, and raw scores.
    """
    df = pd.read_csv(file)
    if 'text' not in df.columns:
        raise ValueError("CSV file must contain a 'text' column.")
    
    df['prediction'] = bino.predict(df['text'].tolist())
    df['raw_score'] = bino.compute_score(df['text'].tolist())
    return df

def batch_interface():
    """
    Create a Gradio interface for uploading CSV files and displaying results.

    Returns:
        gr.Interface: The Gradio interface for batch processing.
    """
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown("## Batch Processing for AI Text Detection")
        with gr.Row():
            csv_input = gr.File(label="Upload CSV", file_types=["csv"])
        with gr.Row():
            output = gr.Dataframe(headers=["text", "prediction", "raw_score"], datatype=["str", "str", "float"])
        with gr.Row():
            download_button = gr.Button("Download Results")
        
        def process_and_display(file):
            df = process_csv(file)
            return df
        
        csv_input.change(process_and_display, inputs=csv_input, outputs=output)
        download_button.click(lambda df: df.to_csv(index=False), inputs=output, outputs=gr.File())
    
    return demo

if __name__ == "__main__":
    # Launch the Gradio interface
    batch_interface().launch()
