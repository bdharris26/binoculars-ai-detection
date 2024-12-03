import gradio as gr
import pandas as pd
import os
from binoculars import Binoculars
import tempfile
from tqdm import tqdm

# Initialize once, not per file
bino = Binoculars()

def process_file(file, batch_size):
    if file is None:
        return None
        
    try:
        if file.name.endswith('.csv'):
            # Process CSV in chunks
            chunks = []
            chunk_iterator = pd.read_csv(file.name, chunksize=batch_size)
            for chunk in tqdm(chunk_iterator, desc="Processing CSV"):
                if 'text' not in chunk.columns:
                    raise gr.Error("Input file must contain a 'text' column")
                chunk['prediction'] = bino.predict(chunk['text'].tolist())
                chunk['raw_score'] = bino.compute_score(chunk['text'].tolist())
                chunks.append(chunk)
            return pd.concat(chunks, ignore_index=True)
        else:
            # Process Excel file in memory with batches
            df = pd.read_excel(file.name)
            if 'text' not in df.columns:
                raise gr.Error("Input file must contain a 'text' column")
            
            predictions = []
            scores = []
            
            for i in tqdm(range(0, len(df), batch_size), desc="Processing Excel"):
                batch = df['text'].iloc[i:i+batch_size].tolist()
                predictions.extend(bino.predict(batch))
                scores.extend(bino.compute_score(batch))
            
            df['prediction'] = predictions
            df['raw_score'] = scores
            return df
            
    except Exception as e:
        raise gr.Error(f"Error processing file: {str(e)}")

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
            batch_size = gr.Slider(
                minimum=1,
                maximum=32,
                value=8,
                step=1,
                label="Batch Size"
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
            inputs=[file_input, batch_size],
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
