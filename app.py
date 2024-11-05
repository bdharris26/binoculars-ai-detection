"""
This file serves as the entry point for launching the Gradio interface for the Binoculars AI text detection demo.
It imports the Gradio app and the compute_raw_score function from the demo module and launches the app when executed.
"""

from demo.demo import app, compute_raw_score

if __name__ == "__main__":
    # Launch the Gradio interface
    app.launch(show_api=False, debug=True, share=True)
