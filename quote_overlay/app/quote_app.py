"""
Quote Overlay Gradio Demo App

Upload an image and enter a quote to generate a professionally designed quote overlay.
"""

import gradio as gr
import argparse
import sys
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch

# Add parent directory to import llava
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import layout_conv
from llava.infer.infer import white_rgb_convert, expand2square


# Global variables
tokenizer = None
model = None
image_processor = None
device = None


def initialize_model(model_path: str, device_name: str = None):
    """Initialize the quote overlay model."""
    global tokenizer, model, image_processor, device

    print(f"Loading model from {model_path}...")

    # Auto-detect device
    if device_name is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_name

    print(f"Using device: {device}")

    # Load model
    with open(Path(model_path) / "adapter_config.json", "r") as f:
        model_base = json.load(f)["base_model_name_or_path"]

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, device=device
    )
    tokenizer.pad_token_id = tokenizer.unk_token_id or 0
    model = model.to(device)

    print("Model loaded successfully!")


def generate_quote_design(image_path: str, quote_text: str, temperature: float = 0.7):
    """
    Generate quote overlay design using the model.

    Args:
        image_path: Path to background image
        quote_text: Quote text to overlay
        temperature: Sampling temperature

    Returns:
        Design parameters dict
    """
    # Load and preprocess image
    image = Image.open(image_path)
    image = white_rgb_convert(image)
    image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
    processed_image = image_processor.preprocess(image, return_tensors="pt", input_data_format="channels_last")["pixel_values"][0]

    # Prepare conversation
    conv = layout_conv.copy()
    conv.sep2 = tokenizer.eos_token

    prompt = (
        f"Given this background image: <image>\n"
        f"Place the following quote aesthetically: '{quote_text}'\n"
        f"Consider the image's color palette, composition, and mood. "
        f"Predict the optimal design parameters."
    )

    conv.append_message("human", prompt)
    conv.append_message("gpt", "")
    prompt_text = conv.get_prompt()

    # Tokenize
    input_ids = tokenizer_image_token(prompt_text, tokenizer, return_tensors="pt").unsqueeze(0).to(device)
    images = [processed_image.to(device, dtype=torch.float16)]
    attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device)

    # Generate
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images,
            attention_mask=attention_mask,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            max_length=2048,
        )

    # Decode
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    print(f"Model output: {output}")

    # Parse JSON
    try:
        design_params = json.loads(output)
        return design_params
    except json.JSONDecodeError:
        print(f"Failed to parse JSON output: {output}")
        # Return default design
        return {
            "text_box": {"left": 0.1, "top": 0.4, "width": 0.8, "height": 0.2},
            "text_style": {
                "font_family": "Arial",
                "font_size": 48,
                "text_color": "#FFFFFF",
                "text_align": "center"
            },
            "background_overlay": {
                "type": "solid_box",
                "color": "#000000",
                "opacity": 0.6
            }
        }


def render_quote_overlay(background_path: str, quote_text: str, design_params: dict):
    """
    Render the quote overlay on the background image.

    Args:
        background_path: Path to background image
        quote_text: Quote text
        design_params: Design parameters from model

    Returns:
        PIL Image with quote overlay
    """
    # Load background
    bg = Image.open(background_path).convert("RGBA")
    width, height = bg.size

    # Create overlay layer
    overlay = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Get text box coordinates
    text_box = design_params["text_box"]
    x = int(text_box["left"] * width)
    y = int(text_box["top"] * height)
    box_width = int(text_box["width"] * width)
    box_height = int(text_box["height"] * height)

    # Draw background overlay if specified
    if "background_overlay" in design_params:
        bg_overlay = design_params["background_overlay"]
        if bg_overlay.get("type") == "solid_box":
            # Parse color
            color = bg_overlay.get("color", "#000000")
            opacity = bg_overlay.get("opacity", 0.6)

            # Convert hex to RGB
            if color.startswith("#"):
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                box_color = (r, g, b, int(opacity * 255))
            else:
                box_color = (0, 0, 0, int(opacity * 255))

            # Draw rounded rectangle
            padding = bg_overlay.get("padding", 20)
            draw.rounded_rectangle(
                [x - padding, y - padding, x + box_width + padding, y + box_height + padding],
                radius=bg_overlay.get("border_radius", 8),
                fill=box_color
            )

    # Draw text (simplified - in production, use proper text wrapping and font loading)
    text_style = design_params.get("text_style", {})
    font_size = text_style.get("font_size", 48)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Parse text color
    text_color = text_style.get("text_color", "#FFFFFF")
    if text_color.startswith("#"):
        r = int(text_color[1:3], 16)
        g = int(text_color[3:5], 16)
        b = int(text_color[5:7], 16)
        color_rgb = (r, g, b, 255)
    else:
        color_rgb = (255, 255, 255, 255)

    # Calculate text position (centered)
    bbox = draw.textbbox((0, 0), quote_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    text_x = x + (box_width - text_width) // 2
    text_y = y + (box_height - text_height) // 2

    # Draw text
    draw.text((text_x, text_y), quote_text, font=font, fill=color_rgb)

    # Composite
    result = Image.alpha_composite(bg, overlay)

    return result.convert("RGB")


def process_quote(image, quote_text, temperature):
    """Gradio processing function."""
    if image is None:
        return None, "Please upload an image first."

    if not quote_text.strip():
        return None, "Please enter a quote."

    try:
        # Save uploaded image temporarily
        temp_path = "/tmp/temp_background.jpg"
        image.save(temp_path)

        # Generate design
        design_params = generate_quote_design(temp_path, quote_text, temperature)

        # Render
        result = render_quote_overlay(temp_path, quote_text, design_params)

        # Format design params for display
        params_text = json.dumps(design_params, indent=2)

        return result, f"Design Parameters:\n{params_text}"

    except Exception as e:
        return None, f"Error: {str(e)}"


def create_demo():
    """Create Gradio interface."""
    title = """
    # 🎨 Quote Overlay Generator

    Upload a background image and enter a quote to generate a professionally designed overlay.
    """

    with gr.Blocks() as demo:
        gr.Markdown(title)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input")
                image_input = gr.Image(type="pil", label="Background Image")
                quote_input = gr.Textbox(
                    lines=3,
                    placeholder="Enter your quote here...",
                    label="Quote Text"
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature (creativity)",
                    info="Higher = more creative, Lower = more conservative"
                )
                generate_btn = gr.Button("Generate Quote Design", variant="primary")

            with gr.Column():
                gr.Markdown("### Output")
                output_image = gr.Image(label="Result")
                params_output = gr.Textbox(
                    lines=15,
                    label="Design Parameters",
                    interactive=False
                )

        generate_btn.click(
            fn=process_quote,
            inputs=[image_input, quote_input, temperature],
            outputs=[output_image, params_output]
        )

        # Examples
        gr.Markdown("### Examples")
        gr.Examples(
            examples=[
                ["The only way to do great work is to love what you do."],
                ["In the middle of difficulty lies opportunity."],
                ["Be yourself; everyone else is already taken."]
            ],
            inputs=[quote_input]
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Quote Overlay Gradio Demo")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/mps/cpu)")
    parser.add_argument("--share", action="store_true",
                       help="Create public share link")
    parser.add_argument("--server_port", type=int, default=7860,
                       help="Server port")
    parser.add_argument("--server_name", type=str, default="127.0.0.1",
                       help="Server name")
    args = parser.parse_args()

    # Initialize model
    initialize_model(args.model_path, args.device)

    # Create and launch demo
    demo = create_demo()

    print("\n" + "=" * 80)
    print("Launching Quote Overlay Demo")
    print("=" * 80)
    print(f"\nAccess the demo at: http://{args.server_name}:{args.server_port}")

    demo.launch(
        share=args.share,
        server_port=args.server_port,
        server_name=args.server_name,
        show_error=True
    )


if __name__ == "__main__":
    main()
