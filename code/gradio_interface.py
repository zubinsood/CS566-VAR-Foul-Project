"""
Gradio Web Interface for MV-Foul Model Visualization

This creates an interactive web interface where you can:
- Select video clips from the dataset
- See model predictions in real-time
- Compare predictions with ground truth labels
- View video frames with overlaid predictions
"""

import os
import argparse
from typing import Dict, Tuple, Optional, List
import json

import torch
import torch.nn.functional as F
import numpy as np
import gradio as gr
from PIL import Image

from mvfoul_dataset import MVFoulDataset
from mvfoul_model import MVFoulBaseline


# Label mappings
SEVERITY_LABELS = {
    0: "No Offence",
    1: "Offence + No Card",
    2: "Offence + Yellow Card",
    3: "Offence + Red Card"
}

SEVERITY_COLORS = {
    0: "#00FF00",  # Green
    1: "#FFA500",  # Orange
    2: "#FFFF00",  # Yellow
    3: "#FF0000",  # Red
}


class ModelInference:
    """Wrapper class to hold model and dataset for Gradio interface."""
    
    def __init__(
        self,
        checkpoint_path: Optional[str],
        data_root: str,
        split: str = "valid",
        num_frames: int = 16,
        img_size: int = 224,
        use_pretrained: bool = True,
    ):
        self.device = self._get_device()
        # Convert to absolute path to avoid path resolution issues
        if not os.path.isabs(data_root):
            # If relative path, resolve it relative to the script's directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            self.data_root = os.path.abspath(os.path.join(project_root, data_root))
        else:
            self.data_root = os.path.abspath(data_root)
        
        # Normalize the path (resolve any .. or . components)
        self.data_root = os.path.normpath(self.data_root)
        
        # Verify the data root exists
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(
                f"Data root directory not found: {self.data_root}\n"
                f"Please check that the path is correct and the data has been downloaded."
            )
        
        # Verify the split directory exists
        split_dir = os.path.join(self.data_root, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                f"Available splits: {[d for d in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, d))]}"
            )
        
        # Verify annotations file exists
        ann_path = os.path.join(split_dir, "annotations.json")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(
                f"annotations.json not found at {ann_path}\n"
                f"Please ensure the dataset has been fully downloaded and extracted."
            )
        
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size
        
        # Build dataset
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
        ])
        
        print(f"[interface] Loading dataset from: {self.data_root}")
        print(f"[interface] Split: {split}")
        print(f"[interface] Annotations file: {ann_path}")
        self.dataset = MVFoulDataset(
            root=self.data_root,
            split=split,
            num_frames=num_frames,
            transform=transform,
        )
        
        # Load model
        num_foul_types = len(self.dataset.foul_type_mapping)
        self.model = MVFoulBaseline(
            num_foul_types=num_foul_types,
            use_pretrained=use_pretrained,
        )
        
        # Resolve checkpoint path
        if checkpoint_path:
            if not os.path.isabs(checkpoint_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(script_dir)
                checkpoint_path = os.path.abspath(os.path.join(project_root, checkpoint_path))
            
            if os.path.exists(checkpoint_path):
                print(f"[interface] Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                print(f"[interface] Checkpoint not found at {checkpoint_path}, using {'pretrained' if use_pretrained else 'random'} initialization")
        else:
            print(f"[interface] No checkpoint provided, using {'pretrained' if use_pretrained else 'random'} initialization")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Create reverse mapping
        self.foul_type_names = {v: k for k, v in self.dataset.foul_type_mapping.items()}
        
        print(f"[interface] Loaded model with {num_foul_types} foul types")
        print(f"[interface] Dataset has {len(self.dataset)} samples")
    
    @staticmethod
    def _get_device():
        """Get the best available device."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _load_video_frames_high_res(self, video_path: str, num_frames: int) -> np.ndarray:
        """Load video frames at original resolution for display."""
        try:
            import av
        except ImportError:
            raise ImportError("pyav is required for video loading")
        
        container = av.open(video_path)
        frames = []
        
        for frame in container.decode(video=0):
            img = frame.to_rgb().to_ndarray()  # (H, W, 3), uint8
            frames.append(img)
        
        container.close()
        
        if len(frames) == 0:
            raise ValueError(f"No frames decoded from {video_path}")
        
        # Sample frames uniformly
        num_total = len(frames)
        indices = np.linspace(0, num_total - 1, num_frames).astype(int)
        frames = [frames[i] for i in indices]
        
        return np.array(frames)  # (T, H, W, C), uint8
    
    def predict(self, sample_idx: int) -> Dict:
        """Run inference on a dataset sample."""
        if sample_idx < 0 or sample_idx >= len(self.dataset):
            return {
                "error": f"Sample index {sample_idx} out of range (0-{len(self.dataset)-1})"
            }
        
        # Load clip and labels (resized for model)
        clip, severity_true, foul_type_true = self.dataset[sample_idx]
        
        # Add batch dimension
        clip_batch = clip.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            severity_logits, foul_type_logits = self.model(clip_batch)
        
        # Get predictions
        severity_pred = severity_logits.argmax(dim=1).item()
        foul_type_pred = foul_type_logits.argmax(dim=1).item()
        
        # Get probabilities
        severity_probs = F.softmax(severity_logits, dim=1)[0].cpu().numpy()
        foul_type_probs = F.softmax(foul_type_logits, dim=1)[0].cpu().numpy()
        
        # Load original resolution frames for display
        sample = self.dataset.samples[sample_idx]
        video_path = self.dataset._get_first_view_path(sample["action_dir"])
        try:
            # Load at original resolution
            frames_high_res = self._load_video_frames_high_res(video_path, self.num_frames)
            frames = [frames_high_res[i] for i in range(len(frames_high_res))]
        except Exception as e:
            print(f"[interface] Warning: Could not load high-res frames, using resized: {e}")
            # Fallback to resized frames
            clip_np = clip.permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C)
            clip_np = np.clip(clip_np, 0, 1)
            frames = [(clip_np[i] * 255).astype(np.uint8) for i in range(len(clip_np))]
        
        return {
            "frames": frames,
            "severity_pred": severity_pred,
            "severity_true": severity_true,
            "foul_type_pred": foul_type_pred,
            "foul_type_true": foul_type_true,
            "severity_probs": severity_probs,
            "foul_type_probs": foul_type_probs,
            "sample_idx": sample_idx,
        }
    
    def get_sample_info(self, sample_idx: int) -> Dict:
        """Get information about a dataset sample."""
        if sample_idx < 0 or sample_idx >= len(self.dataset):
            return {"error": f"Invalid sample index: {sample_idx}"}
        
        sample = self.dataset.samples[sample_idx]
        return {
            "action_id": sample["key"],
            "action_dir": sample["action_dir"],
            "severity": sample["severity"],
            "foul_type": sample["foul_type"],
        }


def create_video_with_overlay(frames: List[np.ndarray], pred_text: str) -> str:
    """Create a video file with text overlay using imageio."""
    if not frames:
        return None
    
    try:
        import imageio
        from PIL import ImageDraw, ImageFont
        import tempfile
        
        # Add text overlay to frames using PIL
        frames_with_text = []
        for frame in frames:
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                try:
                    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
            
            # Draw background rectangle for text
            bbox = draw.textbbox((0, 0), pred_text, font=font)
            padding = 10
            draw.rectangle(
                [(5, 5), (bbox[2] + padding, bbox[3] + padding)],
                fill=(0, 0, 0, 180)
            )
            draw.text((10, 10), pred_text, fill=(255, 255, 255), font=font)
            frames_with_text.append(np.array(img))
        
        # Create video using imageio with H.264 codec
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"inference_video_{os.getpid()}.mp4")
        
        imageio.mimsave(
            temp_path,
            frames_with_text,
            fps=2,
            codec='libx264',
            quality=9,  # Higher quality (0-10 scale, 10 is best)
            pixelformat='yuv420p',  # Ensures web compatibility
            ffmpeg_params=['-crf', '18']  # Lower CRF = higher quality (18 is visually lossless)
        )
        return temp_path
    except ImportError:
        print("[interface] Warning: imageio not available, returning first frame as image")
        # Fallback: return first frame as image
        if frames:
            img = Image.fromarray(frames[0])
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"inference_frame_{os.getpid()}.png")
            img.save(temp_path)
            return temp_path
        return None
    except Exception as e:
        print(f"[interface] Warning: Could not create video with overlay: {e}")
        # Fallback: return first frame as image
        if frames:
            img = Image.fromarray(frames[0])
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"inference_frame_{os.getpid()}.png")
            img.save(temp_path)
            return temp_path
        return None


def format_prediction_display(result: Dict, foul_type_names: Dict[int, str]) -> Tuple[str, str, str, str, str]:
    """Format prediction results for display."""
    if "error" in result:
        return result["error"], "", "", "", ""
    
    severity_pred = result["severity_pred"]
    severity_true = result["severity_true"]
    foul_type_pred = result["foul_type_pred"]
    foul_type_true = result["foul_type_true"]
    severity_probs = result["severity_probs"]
    foul_type_probs = result["foul_type_probs"]
    
    # Format severity prediction
    sev_pred_label = SEVERITY_LABELS[severity_pred]
    sev_true_label = SEVERITY_LABELS[severity_true]
    sev_confidence = severity_probs[severity_pred] * 100
    sev_correct = "✓" if severity_pred == severity_true else "✗"
    sev_color = SEVERITY_COLORS[severity_pred]
    
    severity_html = f"""
    <div style="padding: 10px; border: 2px solid {sev_color}; border-radius: 5px; margin: 5px;">
        <h3>Severity Prediction {sev_correct}</h3>
        <p><b>Predicted:</b> {sev_pred_label} ({sev_confidence:.1f}%)</p>
        <p><b>Ground Truth:</b> {sev_true_label}</p>
        <div style="background: {sev_color}; height: 20px; width: {sev_confidence}%; border-radius: 3px;"></div>
    </div>
    """
    
    # Format foul type prediction
    type_pred_label = foul_type_names.get(foul_type_pred, "Unknown")
    type_true_label = foul_type_names.get(foul_type_true, "Unknown")
    type_confidence = foul_type_probs[foul_type_pred] * 100
    type_correct = "✓" if foul_type_pred == foul_type_true else "✗"
    
    # Get top 3 predictions
    top3_indices = np.argsort(foul_type_probs)[-3:][::-1]
    top3_html = "<ul>"
    for idx in top3_indices:
        name = foul_type_names.get(idx, "Unknown")
        prob = foul_type_probs[idx] * 100
        is_pred = " (PREDICTED)" if idx == foul_type_pred else ""
        top3_html += f"<li>{name}: {prob:.1f}%{is_pred}</li>"
    top3_html += "</ul>"
    
    foul_type_html = f"""
    <div style="padding: 10px; border: 2px solid #333; border-radius: 5px; margin: 5px;">
        <h3>Foul Type Prediction {type_correct}</h3>
        <p><b>Predicted:</b> {type_pred_label} ({type_confidence:.1f}%)</p>
        <p><b>Ground Truth:</b> {type_true_label}</p>
        <h4>Top 3 Predictions:</h4>
        {top3_html}
        <div style="background: #4CAF50; height: 20px; width: {type_confidence}%; border-radius: 3px;"></div>
    </div>
    """
    
    # Summary
    summary = f"Sample {result['sample_idx']}: {sev_correct} Severity, {type_correct} Type"
    
    # Probability distributions
    sev_dist = ", ".join([f"{SEVERITY_LABELS[i]}: {severity_probs[i]*100:.1f}%" for i in range(4)])
    type_dist = ", ".join([f"{foul_type_names.get(i, 'Unknown')}: {foul_type_probs[i]*100:.1f}%" 
                          for i in range(min(5, len(foul_type_probs)))])
    
    return summary, severity_html, foul_type_html, sev_dist, type_dist


def predict_sample(sample_idx: int) -> Tuple:
    """Gradio prediction function."""
    try:
        sample_idx = int(sample_idx)
        result = inference.predict(sample_idx)
        
        if "error" in result:
            return None, [], result["error"], "", "", ""
        
        # Create video with overlay
        pred_text = f"Pred: {SEVERITY_LABELS[result['severity_pred']]} | {inference.foul_type_names.get(result['foul_type_pred'], 'Unknown')}"
        video_path = create_video_with_overlay(result["frames"], pred_text)
        
        # If video creation failed, try alternative: return frames as images
        if video_path is None or not os.path.exists(video_path):
            # Fallback: create a simple video using imageio if available
            try:
                import imageio
                temp_dir = os.path.join(os.path.dirname(__file__), "..")
                video_path = os.path.join(temp_dir, "temp_inference_video.mp4")
                
                # Add text to frames using PIL
                frames_with_text = []
                for frame in result["frames"]:
                    img = Image.fromarray(frame)
                    # Add text overlay using PIL
                    from PIL import ImageDraw, ImageFont
                    draw = ImageDraw.Draw(img)
                    try:
                        font = ImageFont.truetype("arial.ttf", 20)
                    except:
                        font = ImageFont.load_default()
                    # Draw background rectangle
                    bbox = draw.textbbox((0, 0), pred_text, font=font)
                    draw.rectangle([(5, 5), (bbox[2] + 10, bbox[3] + 10)], fill=(0, 0, 0))
                    draw.text((10, 10), pred_text, fill=(255, 255, 255), font=font)
                    frames_with_text.append(np.array(img))
                
                imageio.mimsave(video_path, frames_with_text, fps=2, codec='libx264')
            except ImportError:
                # Last resort: return first frame as image
                if result["frames"]:
                    img = Image.fromarray(result["frames"][0])
                    temp_dir = os.path.join(os.path.dirname(__file__), "..")
                    video_path = os.path.join(temp_dir, "temp_inference_frame.png")
                    img.save(video_path)
        
        # Format display
        summary, sev_html, type_html, sev_dist, type_dist = format_prediction_display(result, inference.foul_type_names)
        
        # Also prepare frames for gallery display
        frame_images = [(Image.fromarray(frame), f"Frame {i+1}") for i, frame in enumerate(result["frames"])]
        
        return video_path, frame_images, summary, sev_html, type_html, f"Severity: {sev_dist}\nType: {type_dist}"
    
    except Exception as e:
        import traceback
        return None, [], f"Error: {str(e)}\n{traceback.format_exc()}", "", "", ""


def get_random_sample() -> int:
    """Get a random sample index."""
    import random
    return random.randint(0, len(inference.dataset) - 1)


# Global inference object (will be initialized in main)
inference = None


def main():
    global inference
    
    parser = argparse.ArgumentParser(description="Gradio interface for MV-Foul model")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to mvfouls root (folder containing train/ and valid/)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        choices=["train", "valid"],
        help="Dataset split to use",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames per clip",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Image size for model input",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run Gradio interface",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Don't use pretrained ResNet weights",
    )
    
    args = parser.parse_args()
    
    print("[interface] Initializing model and dataset...")
    inference = ModelInference(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        split=args.split,
        num_frames=args.num_frames,
        img_size=args.img_size,
        use_pretrained=not args.no_pretrained,
    )
    
    # Create Gradio interface
    with gr.Blocks(title="VAR Foul Recognition - Model Visualization", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ⚽ VAR Foul Recognition - Model Visualization")
        gr.Markdown("Select a video clip from the dataset to see the model's predictions for foul severity and type.")
        
        with gr.Row():
            with gr.Column(scale=1):
                sample_slider = gr.Slider(
                    minimum=0,
                    maximum=len(inference.dataset) - 1,
                    value=0,
                    step=1,
                    label="Sample Index",
                    info=f"Select a sample (0-{len(inference.dataset)-1})"
                )
                
                with gr.Row():
                    random_btn = gr.Button("🎲 Random Sample", variant="secondary")
                    predict_btn = gr.Button("🔍 Predict", variant="primary")
            
            with gr.Column(scale=2):
                video_output = gr.Video(label="Video Clip with Predictions", format="mp4")
                frame_gallery = gr.Gallery(label="Video Frames", show_label=True, elem_id="gallery", 
                                          columns=4, rows=2, height="auto")
        
        with gr.Row():
            summary_output = gr.Textbox(label="Summary", interactive=False)
        
        with gr.Row():
            with gr.Column():
                severity_output = gr.HTML(label="Severity Prediction")
            with gr.Column():
                foul_type_output = gr.HTML(label="Foul Type Prediction")
        
        prob_output = gr.Textbox(label="Probability Distributions", lines=3, interactive=False)
        
        # Event handlers
        predict_btn.click(
            fn=predict_sample,
            inputs=[sample_slider],
            outputs=[video_output, frame_gallery, summary_output, severity_output, foul_type_output, prob_output]
        )
        
        random_btn.click(
            fn=get_random_sample,
            outputs=[sample_slider]
        ).then(
            fn=predict_sample,
            inputs=[sample_slider],
            outputs=[video_output, frame_gallery, summary_output, severity_output, foul_type_output, prob_output]
        )
        
        sample_slider.change(
            fn=predict_sample,
            inputs=[sample_slider],
            outputs=[video_output, frame_gallery, summary_output, severity_output, foul_type_output, prob_output]
        )
        
        # Auto-predict on load
        demo.load(
            fn=predict_sample,
            inputs=[sample_slider],
            outputs=[video_output, frame_gallery, summary_output, severity_output, foul_type_output, prob_output]
        )
    
    print(f"[interface] Launching Gradio interface on port {args.port}...")
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

