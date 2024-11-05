from demo.demo import app, compute_raw_score

if __name__ == "__main__":
    # Launch the Gradio interface
    app.launch(show_api=False, debug=True, share=True)
