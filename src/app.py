import os
import base64
import gradio as gr

# Internal Modular Imports
from src.config import LOGO_PATH
from src.chain import get_rag_chain

# =====================================================
# INITIALIZATION
# =====================================================
rag_chain = get_rag_chain()

def chat_rag(message, history):

    # print("USER QUESTION:", message)
    """
    history: List of dictionaries like [{"role": "user", "content": "hello"}]
    message: The new string from the user
    """
    if not message.strip():
        return history

    try:
        # We pass the message to our modular chain
        answer = rag_chain.invoke(message)
        # print("CHAIN ANSWER:", answer)
    except Exception as e:
        answer = f"⚠️ **Error:** {str(e)}"
        # answer = f"ERROR: {e}"

    # Append the new interaction to the history list
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    # history.append([message, answer])
    
    return history
    

# =====================================================
# LOGO & HEADER LOGIC
# =====================================================
def encode_image(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")

logo_b64 = encode_image(LOGO_PATH)

header_html = f"""
<div style="display: flex; align-items: center; gap: 14px; padding: 12px 0;">
    <img src="data:image/jpeg;base64,{logo_b64}" style="height: 50px; object-fit: contain;" />
    <span style="font-size: 22px; font-weight: 600;">Thermo Med Assistant</span>
</div>
"""

# =====================================================
# GRADIO UI (Blocks API)
# =====================================================
with gr.Blocks(css="""
    .gr-chatbot { border-radius: 12px !important; }
    .message-wrap { font-size: 15px !important; }
""") as interface:

    gr.HTML(header_html)
    gr.Markdown("Ask **genetic-based questions** related to **ThermoFisher Scientific**.")

    # 1. THE CHATBOT COMPONENT (Stores history)
    chatbot = gr.Chatbot(
        height=450,
        show_label=False
    )

    # 2. THE INPUT BOX
    question_input = gr.Textbox(
        placeholder="Ask any query related to ThermoFisher Scientific...",
        lines=1,
        container=False
    )

    # 3. BUTTONS
    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        clear_btn = gr.Button("Clear Chat")

    # =====================================================
    # EVENT LISTENERS
    # =====================================================
    
    # When user clicks Submit or hits Enter
    submit_event = submit_btn.click(
        fn=chat_rag,
        inputs=[question_input, chatbot],
        outputs=chatbot
    ).then(
        fn=lambda: "",  # This clears the textbox after sending
        outputs=question_input
    )

    # Clear button logic
    clear_btn.click(
        fn=lambda: [],
        outputs=chatbot
    )

if __name__ == "__main__":
    interface.launch(server_name="127.0.0.1", server_port=7860)