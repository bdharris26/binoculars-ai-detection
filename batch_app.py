import gradio as gr
import pandas as pd
import os
from binoculars import Binoculars
import tempfile
from tqdm import tqdm
import plotly.express as px

# Initialize once, not per file
bino = Binoculars()

def create_score_plot(df):
    if df is None or len(df) == 0:
        return None
    fig = px.histogram(df, x="raw_score",
                       title="Distribution of AI Detection Scores",
                       labels={"raw_score": "AI Detection Score"},
                       nbins=20)
    fig.update_layout(showlegend=False)
    return fig

def process_file(file, batch_size):
    if file is None:
        return None, None

    try:
        # Determine file type
        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext == '.csv':
            df_iterator = pd.read_csv(file.name, chunksize=batch_size)
        elif file_ext == '.xlsx':
            df = pd.read_excel(file.name)
            # Simulate an iterator for uniformity
            df_iterator = (df[i:i+batch_size] for i in range(0, len(df), batch_size))
        else:
            raise gr.Error("Unsupported file type. Please upload a CSV or XLSX file.")

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

    except Exception as e:
        raise gr.Error(f"Error processing file: {e}")

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
                visible=True,
                wrap=True,
                column_widths=["65%", "20%", "15%"],
                elem_classes=["fixed-height-row"]
            )
        with gr.Row():
            plot = gr.Plot(label="Score Distribution")
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

        # Update existing file_input change handler to include plot update
        file_input.change(
            fn=process_file,
            inputs=[file_input, batch_size],
            outputs=[output, plot]
        )

        # Add plot update when dataframe changes
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
