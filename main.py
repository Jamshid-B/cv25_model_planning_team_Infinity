import gradio as gr
from think import encode_image, analyze_image_with_query

def process_input(user_text, image_file):
    try:
        if image_file is not None:
            encoded_image = encode_image(image_file.name if hasattr(image_file, 'name') else image_file)
            response = analyze_image_with_query(user_text, encoded_image)
        else:
            response = "Please upload an image."
        return response
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks() as iface:
    gr.Markdown("# AI Medical Bot")
    gr.Markdown("Upload an image and ask a medical question")
    
    with gr.Row():
        question = gr.Textbox(label="Enter your question", lines=3)
        image_input = gr.Image(label="Upload medical image", type="filepath")
    
    submit_btn = gr.Button("Analyze", variant="primary")
    output = gr.Textbox(label="Response", lines=5)
    
    submit_btn.click(
        fn=process_input,
        inputs=[question, image_input],
        outputs=output
    )

if __name__ == "__main__":
    iface.launch(share=False, show_api=False)