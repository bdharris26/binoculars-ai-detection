import gradio as gr
import pandas as pd
import os
from binoculars import Binoculars
import tempfile

# Initialize the Binoculars model
bino = Binoculars()

def process_file(file):
    try:
        # Check if file is None or empty
        if file is None:
            raise ValueError("No file uploaded")
            
        # Check file extension
        filename = file.name
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(file.name)
        elif filename.lower().endswith('.xlsx'):
            df = pd.read_excel(file.name)
        else:
            raise ValueError("File must be a CSV or XLSX")

        # Read and validate file
        if 'text' not in df.columns:
            raise ValueError("File must contain a 'text' column")
        
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
            file_input = gr.File(
                label="Upload CSV or XLSX",
                file_types=[".csv", ".xlsx"],
                type="filepath"
            )
        with gr.Row():
            output = gr.Dataframe(
                headers=["text", "prediction", "raw_score"],
                datatype=["str", "str", "number"],
                visible=True
            )
        with gr.Row():
            download_button = gr.Button("Download Results")
        
        def save_df(df): 
            if df is None:
                return None
                
            # Create temporary file
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, "results.xlsx")
            
            try:
                # Always save as xlsx since it preserves data types better
                df.to_excel(output_path, index=False)
                return output_path
            except Exception as e:
                raise gr.Error(f"Error saving results: {str(e)}")
        
        file_input.change(
            fn=process_file,
            inputs=file_input,
            outputs=output
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
