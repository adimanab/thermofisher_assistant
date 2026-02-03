import gradio as gr
import base64
import os
from config import LOGO_PATH, validate_config
from vector_store import initialize_vector_db
from rag_engine import rag_query

# =====================================================
# INITIALIZATION
# =====================================================
validate_config()
initialize_vector_db()

# =====================================================
# UI HELPERS
# =====================================================
def encode_image(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")

logo_b64 = encode_image(LOGO_PATH)

header_html = f"""
<div style="
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 0;
">
    <img
        src="data:image/jpeg;base64,{logo_b64}"
        style="height: 50px; object-fit: contain;"
    />
    <span style="font-size: 22px; font-weight: 600;">
        Thermo Med Assistant
    </span>
</div>
"""


def chat_rag(message, history):
    if not message.strip():
        return history

    answer = rag_query(message)

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer}
    ]

    return history

# =====================================================
# GRADIO UI
# =====================================================
with gr.Blocks(
   css="""
    /* ================== GLOBAL LAYOUT ================== */
    html, body, .gradio-container {
        height: 100%;
        margin: 0;
    }

    .gradio-container {
        display: flex;
        justify-content: center;
        background-color: var(--background-fill-primary);
    }

    #root, .app {
        width: 100%;
        max-width: 1100px;
        height: 100%;
        display: flex;
        flex-direction: column;
        padding: 16px;
    }

    .gr-chatbot {
        flex: 1;
        overflow-y: auto;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #374151;
    }

    .gr-chat-message {
        max-width: 75%;
        padding: 12px 14px;
        border-radius: 14px;
        font-size: 14.5px;
        line-height: 1.5;
        word-wrap: break-word;
    }

    .gr-chat-message.user {
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }

    .gr-chat-message.bot {
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }

    #input-bar {
        position: sticky;
        bottom: 0;
        background-color: var(--background-fill-primary);
        padding-top: 12px;
    }

    textarea {
        border-radius: 12px !important;
        padding: 12px !important;
        font-size: 14.5px !important;
        resize: none !important;
    }

    textarea::placeholder {
        color: #9ca3af;
    }

    button {
        border-radius: 10px !important;
    }

    @media (prefers-color-scheme: light) {
        .gr-chatbot {
            background-color: #ffffff;
        }

        .gr-chat-message {
            background-color: #ffffff;
            color: #111827;
            border: 1px solid #e5e7eb;
        }

        .gr-chat-message.user {
            background-color: #e0f2fe;
            border-color: #bae6fd;
        }
    }

    @media (prefers-color-scheme: dark) {
        .gr-chatbot {
            background-color: #020617;
        }

        .gr-chat-message {
            background-color: #111827;
            color: #e5e7eb;
            border: 1px solid #374151;
        }

        .gr-chat-message.user {
            background-color: #1e293b;
            border-color: #334155;
        }

        footer {
    display: none !important;
   }
    }
    """
) as interface:

    # Header
    gr.HTML(header_html)

    gr.Markdown(
        "Ask **genetic-based questions** related to **ThermoFisher Scientific**."
    )

    # Chat window
    chatbot = gr.Chatbot(
        height=420,
        show_label=False
    )

    # Input
    question_input = gr.Textbox(
        placeholder="Ask anything about ThermoFisher Scientific...",
        lines=1,
        max_lines=4,
        show_label=False,
        label=None
    )

    # Buttons
    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear Chat")

    # Send message
    send_btn.click(
        fn=chat_rag,
        inputs=[question_input, chatbot],
        outputs=chatbot
    )

    # Clear input after send
    send_btn.click(
        fn=lambda: "",
        inputs=None,
        outputs=question_input
    )

    # Clear chat
    clear_btn.click(
        fn=lambda: [],
        inputs=None,
        outputs=chatbot
    )

if __name__ == "__main__":
    interface.launch(
    server_name="0.0.0.0",
    server_port=7860
)
