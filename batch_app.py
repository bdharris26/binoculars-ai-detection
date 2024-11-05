import gradio as gr
import pandas as pd
import os
from binoculars import Binoculars

# Initialize the Binoculars model
bino = Binoculars()

def process_csv(file):
    try:
        # Check if file is None or empty
        if file is None:
            raise ValueError("No file uploaded")
            
        # Check file extension
        filename = file.name
        if not filename.lower().endswith('.csv'):
            raise ValueError("File must be a CSV")

        # Read and validate CSV
        df = pd.read_csv(file.name)
        if 'text' not in df.columns:
            raise ValueError("CSV file must contain a 'text' column")
        
        # Process the text
        df['prediction'] = bino.predict(df['text'].tolist())
        df['raw_score'] = bino.compute_score(df['text'].tolist())
        return df
    except Exception as e:
        raise gr.Error(str(e))

def batch_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown("## Batch Processing for AI Text Detection")
        with gr.Row():
            # Fix: Changed type to 'filepath' instead of 'file'
            csv_input = gr.File(
                label="Upload CSV", 
                file_types=["csv"],  # Removed dot prefix
                type="filepath"      # Changed from 'file' to 'filepath'
            )
        with gr.Row():
            output = gr.Dataframe(
                headers=["text", "prediction", "raw_score"],
                datatype=["str", "str", "number"],
                visible=True
            )
        with gr.Row():
            download_button = gr.Button("Download Results")
        
        # Update event handlers
        csv_input.change(
            fn=process_csv,
            inputs=csv_input,
            outputs=output
        )
        
        download_button.click(
            fn=lambda df: df.to_csv(index=False) if df is not None else None,
            inputs=output,
            outputs=gr.File(label="Download CSV")
        )
    
    return demo

if __name__ == "__main__":
    batch_interface().launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )