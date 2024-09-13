import os,sys
import torch
import folder_paths
from huggingface_hub import snapshot_download
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)
ckpt_dir = os.path.join(now_dir,"pretrained_models")
pretrained_model_path = os.path.join(ckpt_dir,"rv-5-1")
pretrained_clip_path = os.path.join(ckpt_dir,"dinov2")
unet_checkpoint_path = os.path.join(ckpt_dir,"realisdance")
output_dir = folder_paths.get_output_directory()

import cv2
import pickle
import decord
import numpy as np
from decord import VideoReader
from decord.bridge.torchdl import to_torch
from torchvision.transforms import transforms
from transformers import AutoModel
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler, AutoencoderKLTemporalDecoder
from realisdance.data.dwpose_utils.draw_pose import draw_pose
from realisdance.models.rd_unet import RealisDanceUnet
from realisdance.pipelines.pipeline import RealisDancePipeline
from realisdance.utils.util import save_videos_grid

decord.bridge.set_bridge('torch')

def augmentation(frame, transform, state=None):
    if state is not None:
        torch.set_rng_state(state)
    return transform(frame)

def simple_reader(ref_image, dwpose_path, hamer_path, smpl_path, sample_size, clip_size, max_length):
    scale = (1.0, 1.0)
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        # ratio is w/h
        transforms.RandomResizedCrop(
            sample_size, scale=scale,
            ratio=(sample_size[1] / sample_size[0], sample_size[1] / sample_size[0]), antialias=True),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    clip_transform = transforms.Compose([
        transforms.ToTensor(),
        # ratio is w/h
        transforms.RandomResizedCrop(
            clip_size, scale=scale,
            ratio=(clip_size[1] / clip_size[0], clip_size[1] / clip_size[0]), antialias=True),
        transforms.Normalize([0.485, 0.456, 0.406],  # used for dino
                             [0.229, 0.224, 0.225],  # used for dino
                             inplace=True),
    ])
    pose_transform = transforms.Compose([
        # ratio is w/h
        transforms.RandomResizedCrop(
            sample_size, scale=scale,
            ratio=(sample_size[1] / sample_size[0], sample_size[1] / sample_size[0]), antialias=True),
    ])

    hamer_reader = VideoReader(hamer_path)
    smpl_reader = VideoReader(smpl_path)
    with open(dwpose_path, 'rb') as pose_file:
        pose_list = pickle.load(pose_file)
    assert len(hamer_reader) == len(smpl_reader) == len(pose_list)
    video_length = len(hamer_reader)
    batch_index = range(0, video_length, 4)[:max_length]

    hamer = to_torch(hamer_reader.get_batch(batch_index)).permute(0, 3, 1, 2).contiguous() / 255.0
    smpl = to_torch(smpl_reader.get_batch(batch_index)).permute(0, 3, 1, 2).contiguous() / 255.0

    pose = [draw_pose(pose_list[batch_index[idx]], hamer.shape[-2], hamer.shape[-1], draw_face=False)
            for idx in range(len(batch_index))]
    pose = torch.from_numpy(
        np.stack(pose, axis=0)).permute(0, 3, 1, 2).contiguous() / 255.0
    
    img_np = ref_image.numpy()[0] * 255
    _ref_img = img_np.astype(np.uint8)
    #_ref_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    state = torch.get_rng_state()
    ref_image = augmentation(_ref_img, img_transform, state)
    ref_image_clip = augmentation(_ref_img, clip_transform, state)
    pose = augmentation(pose, pose_transform, state)
    hamer = augmentation(hamer, pose_transform, state)
    smpl = augmentation(smpl, pose_transform, state)

    del hamer_reader
    del smpl_reader
    return (
        ref_image.unsqueeze(0),
        ref_image_clip.unsqueeze(0),
        pose.permute(1, 0, 2, 3).unsqueeze(0).contiguous(),
        hamer.permute(1, 0, 2, 3).unsqueeze(0).contiguous(),
        smpl.permute(1, 0, 2, 3).unsqueeze(0).contiguous(),
    )


class RealisDanceNode:
    def __init__(self):
        if not os.path.exists(os.path.join(pretrained_model_path,"unet","diffusion_pytorch_model.safetensors")):
            snapshot_download(repo_id="SG161222/Realistic_Vision_V5.1_noVAE",
                            local_dir=pretrained_model_path,
                            ignore_patterns=["Realistic*"])
        if not os.path.exists(os.path.join(pretrained_clip_path,"model.safetensors")):
            snapshot_download(repo_id="facebook/dinov2-large",
                              local_dir=pretrained_clip_path,
                              ignore_patterns=["*.bin"])
        if not os.path.exists(os.path.join(unet_checkpoint_path,"stage_2_hamer_release.ckpt")):
            snapshot_download(repo_id="theFoxofSky/RealisDance",
                              local_dir=unet_checkpoint_path)
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "ref_image":("IMAGE",),
                "dwpose_path":("PATH",),
                "hamer_path":("PATH",),
                "smpl_path":("PATH",),
                "fps":("INT",{
                    "default":8,
                }),
                "mixed_precision":(["fp16", "bf16"],),
                "seed":("INT",{
                    "default":42,
                })
            }
        }

    RETURN_TYPES = ("VIDEO","VIDEO",)
    RETURN_NAMES = ("sample_video","refer_sequnces",)

    FUNCTION = "gen_video"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_RealisDance"

    def gen_video(self,ref_image,dwpose_path,hamer_path,smpl_path,fps,
                  mixed_precision,seed):
        torch.manual_seed(seed)
        # Load scheduler, tokenizer and models
        print("Load scheduler, tokenizer and models.")
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
        image_encoder = AutoModel.from_pretrained(pretrained_clip_path)
        noise_scheduler_kwargs_dict = {}
        config = OmegaConf.load(os.path.join(now_dir,"configs","stage2_hamer.yaml"))
        if config['zero_snr']:

            print("Enable Zero-SNR")
            noise_scheduler_kwargs_dict["rescale_betas_zero_snr"] = True
            if config["v_pred"]:
                noise_scheduler_kwargs_dict["prediction_type"] = "v_prediction"
                noise_scheduler_kwargs_dict["timestep_spacing"] = "linspace"
        noise_scheduler = DDIMScheduler.from_pretrained(
            pretrained_model_path,
            subfolder="scheduler",
            **noise_scheduler_kwargs_dict,
        )

        unet = RealisDanceUnet(
            pretrained_model_path=pretrained_model_path,
            image_finetune=False,
            unet_additional_kwargs=config["unet_additional_kwargs"],
            pose_guider_kwargs=config["pose_guider_kwargs"],
            clip_projector_kwargs=config["clip_projector_kwargs"],
            fix_ref_t=config["fix_ref_t"],
            fusion_blocks="full",
        )
        # Load pretrained unet weights
        unet_checkpoint_path = os.path.join(ckpt_dir,"realisdance")
        unet_checkpoint_path = os.path.join(unet_checkpoint_path,"stage_2_hamer_release.ckpt")
        print(f"from checkpoint: {unet_checkpoint_path}")
        unet_checkpoint_path = torch.load(unet_checkpoint_path, map_location="cpu")
        if "global_step" in unet_checkpoint_path:
            print(f"global_step: {unet_checkpoint_path['global_step']}")
        state_dict = unet_checkpoint_path["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_k = k[7:]
            else:
                new_k = k
            new_state_dict[new_k] = state_dict[k]
        m, u = unet.load_state_dict(new_state_dict, strict=False)
        print(f"Load from checkpoint with missing keys:\n{m}")
        print(f"Load from checkpoint with unexpected keys:\n{u}")

        # Freeze vae and image_encoder
        vae.eval()
        vae.requires_grad_(False)
        image_encoder.eval()
        image_encoder.requires_grad_(False)
        unet.eval()
        unet.requires_grad_(False)

        # Set validation pipeline
        validation_pipeline = RealisDancePipeline(
            unet=unet, vae=vae, image_encoder=image_encoder, scheduler=noise_scheduler)
        validation_pipeline.image_finetune = False
        validation_kwargs_container = {} if config["validation_kwargs"] is None else OmegaConf.to_container(config["validation_kwargs"])
        if config["vae_slicing"] and 'SVD' not in pretrained_model_path:
            validation_pipeline.enable_vae_slicing()

        # move to cuda
        vae.to("cuda")
        image_encoder.to("cuda")
        unet.to("cuda")
        validation_pipeline = validation_pipeline.to("cuda")
        sample_size = (768,576)
        val_ref_image, val_ref_image_clip, val_pose, val_hamer, val_smpl = simple_reader(
            ref_image=ref_image,
            dwpose_path=dwpose_path,
            hamer_path=hamer_path,
            smpl_path=smpl_path,
            sample_size=sample_size,
            clip_size=(320,240),
            max_length=80,
        )
        print("***** Running validation *****")
        generator = torch.Generator(device=unet.device)
        generator.manual_seed(seed)

        height, width = sample_size

        val_ref_image = val_ref_image.to("cuda")
        val_ref_image_clip = val_ref_image_clip.to("cuda")
        val_pose = val_pose.to("cuda")
        val_hamer = val_hamer.to("cuda")
        val_smpl = val_smpl.to("cuda")

        # Predict the noise residual and compute loss
        # Mixed-precision training
        if mixed_precision in ("fp16", "bf16"):
            weight_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
        else:
            weight_dtype = torch.float32
        with torch.cuda.amp.autocast(
            enabled=mixed_precision in ("fp16", "bf16"),
            dtype=weight_dtype
        ):
            sample = validation_pipeline(
                pose=val_pose,
                hamer=val_hamer,
                smpl=val_smpl,
                ref_image=val_ref_image,
                ref_image_clip=val_ref_image_clip,
                height=height, width=width,
                fake_uncond=not config["train_cfg"],
                **validation_kwargs_container).videos

        video_length = sample.shape[2]
        val_ref_image = val_ref_image.unsqueeze(2).repeat(1, 1, video_length, 1, 1)
        save_obj = torch.cat([
            (val_ref_image.cpu() / 2 + 0.5).clamp(0, 1),
            val_pose.cpu(),
            val_hamer.cpu(),
            val_smpl.cpu(),
            # sample.cpu(),
        ], dim=-1)
        dwpose_name = os.path.splitext(os.path.basename(dwpose_path))[0]
        hamer_name = os.path.splitext(os.path.basename(hamer_path))[0]
        smpl_name = os.path.splitext(os.path.basename(smpl_path))[0]
        output_ref_name = f"d_{dwpose_name}_h_{hamer_name}_s_{smpl_name}"

        sample_path = os.path.join(output_dir,f"sample_{output_ref_name}_.mp4")
        save_videos_grid(sample.cpu(),sample_path,fps=fps)
        ref_path = os.path.join(output_dir,f"{output_ref_name}.mp4")
        save_videos_grid(save_obj, ref_path, fps=fps)
        return (sample_path, ref_path,)

WEB_DIRECTORY = "./web"
from .util_nodes import LoadFile,LoadVideo,PreViewVideo

NODE_CLASS_MAPPINGS = {
    "LoadFile":LoadFile,
    "LoadVideo":LoadVideo,
    "PreViewVideo":PreViewVideo,
    "RealisDanceNode": RealisDanceNode
}










