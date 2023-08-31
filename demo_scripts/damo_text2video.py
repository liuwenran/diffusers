import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float32, variant="fp32")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.enable_model_cpu_offload()

prompt = "A handsome man is playing basketball"
video_frames = pipe(prompt, num_inference_steps=25).frames
video_path = export_to_video(video_frames, output_video_path='results/damo_text2video/basketball_result2.mp4')
