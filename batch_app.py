import gradio as gr
import pandas as pd
from binoculars import Binoculars

# Initialize the Binoculars model
bino = Binoculars()

def process_csv(file):
    try:
        df = pd.read_csv(file)
        if 'text' not in df.columns:
            raise ValueError("CSV file must contain a 'text' column.")
        
        df['prediction'] = bino.predict(df['text'].tolist())
        df['raw_score'] = bino.compute_score(df['text'].tolist())
        return df
    except Exception as e:
        raise gr.Error(f"Error processing file: {str(e)}")

def batch_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown("## Batch Processing for AI Text Detection")
        with gr.Row():
            csv_input = gr.File(label="Upload CSV", file_types=["csv"])
        with gr.Row():
            # Fix: Changed datatype parameter to correct types
            output = gr.Dataframe(
                headers=["text", "prediction", "raw_score"],
                datatype=["str", "str", "number"]  # Changed 'float' to 'number'
            )
        with gr.Row():
            download_button = gr.Button("Download Results")
        
        def process_and_display(file):
            if file is None:
                return None
            return process_csv(file)
        
        csv_input.change(process_and_display, inputs=csv_input, outputs=output)
        download_button.click(
            lambda df: df.to_csv(index=False) if df is not None else None,
            inputs=output,
            outputs=gr.File(label="Download CSV")
        )
    
    return demo

if __name__ == "__main__":
    # Launch the Gradio interface with custom configurations
    batch_interface().launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )