

<div align="center">
    <h1> <a>Paint3D: Paint Anything 3D with Lighting-Less Texture Diffusion Models</a></h1>

<p align="center">
  <a href=https://paint3d.github.io/>Project Page</a> •
  <a href=https://arxiv.org/abs/2312.13913>Arxiv</a> •
  Demo •
  <a href="#️-faq">FAQ</a> •
  <a href="#-citation">Citation</a>
</p>

</div>


https://github.com/OpenTexture/Paint3D/assets/18525299/9aef7eeb-a783-482c-87d5-78055da3bfc0


##  Introduction

Paint3D is a novel coarse-to-fine generative framework that is capable of producing high-resolution, lighting-less, and diverse 2K UV texture maps for untextured 3D meshes conditioned on text or image inputs.

<details open="open">
    <summary><b>Technical details</b></summary>

We present Paint3D, a novel coarse-to-fine generative framework that is capable of producing high-resolution, lighting-less, and diverse 2K UV texture maps for untextured 3D meshes conditioned on text or image inputs. The key challenge addressed is generating high-quality textures without embedded illumination information, which allows the textures to be re-lighted or re-edited within modern graphics pipelines. To achieve this, our method first leverages a pre-trained depth-aware 2D diffusion model to generate view-conditional images and perform multi-view texture fusion, producing an initial coarse texture map. However, as 2D models cannot fully represent 3D shapes and disable lighting effects, the coarse texture map exhibits incomplete areas and illumination artifacts. To resolve this, we train separate UV Inpainting and UVHD diffusion models specialized for the shape-aware refinement of incomplete areas and the removal of illumination artifacts. Through this coarse-to-fine process, Paint3D can produce high-quality 2K UV textures that maintain semantic consistency while being lighting-less, significantly advancing the state-of-the-art in texturing 3D objects.

<img width="1194" alt="pipeline" src="./assets/images/pipeline.jpg">
</details>

## 🚩 News
- [2024/04/26] Upload code 🔥🔥🔥
- [2023/12/21] Upload paper and init project 🔥🔥🔥

## ⚡ Quick Start
### Setup
The code is tested on Centos 7 with PyTorch 1.12.1 CUDA 11.6 installed. Please follow the following steps to setup environment.
```
# install python environment
conda env create -f environment.yaml

# install pytorch
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia

# install kaolin
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu121.html
```


### Txt condition
For UV-position controlnet, you can find it [here](https://huggingface.co/GeorgeQi/Paint3d_UVPos_Control).

To use the other ControlNet models, please download it from the [hugging face page](https://huggingface.co/lllyasviel), and modify the controlnet path in the config file.


Then, you can generate coarse texture via:
```
python pipeline_paint3d_stage1.py  --sd_config controlnet/config/depth_based_inpaint_template.yaml  --render_config paint3d/config/train_config_paint3d.py  --mesh_path data/MOW/data/watertight_models/board_food_v_LUS1jeTGc68_frame000082.obj --texture_path data/MOW/data/images/board_food_v_LUS1jeTGc68_frame000082.jpg  --outdir outputs/board_food_v_LUS1jeTGc68_frame000082_stage1
```

and the refined texture via:
```
python pipeline_paint3d_stage2.py --sd_config controlnet/config/UV_based_inpaint_template.yaml  --render_config paint3d/config/train_config_paint3d.py  --mesh_path data/MOW/data/watertight_models/board_food_v_LUS1jeTGc68_frame000082.obj --texture_path outputs/board_food_v_LUS1jeTGc68_frame000082_stage1/res-0/albedo.png  --outdir outputs/board_food_v_LUS1jeTGc68_frame000082_stage2
```


Optionally, you can also generate texture results with UV position controlnet only, for example:
```
python pipeline_UV_only.py \
 --sd_config controlnet/config/UV_gen_template.yaml \
 --render_config paint3d/config/train_config_paint3d.py \
 --mesh_path demo/objs/teapot/scene.obj \
 --outdir outputs/test_teapot
```


### Image condition

With a image condition, you can generate coarse texture via:
```
python pipeline_paint3d_stage1.py \
 --sd_config controlnet/config/depth_based_inpaint_template.yaml \
 --render_config paint3d/config/train_config_paint3d.py \
 --mesh_path demo/objs/Suzanne_monkey/Suzanne_monkey.obj \
 --prompt " " \
 --ip_adapter_image_path demo/objs/Suzanne_monkey/img_prompt.png \
 --outdir outputs/img_stage1
```

and the refined texture via:
```
python pipeline_paint3d_stage2.py \
--sd_config controlnet/config/UV_based_inpaint_template.yaml \
--render_config paint3d/config/train_config_paint3d.py \
--mesh_path demo/objs/Suzanne_monkey/Suzanne_monkey.obj \
--texture_path outputs/img_stage1/res-0/albedo.png \
--prompt " " \
 --ip_adapter_image_path demo/objs/Suzanne_monkey/img_prompt.png \
--outdir outputs/img_stage2
```



### Model Converting
For checkpoints in [Civitai](https://civitai.com/) with only a .safetensor file, you can use the following script to convert and use them. 
```
python tools/convert_original_stable_diffusion_to_diffusers.py \
--checkpoint_path YOUR_LOCAL.safetensors \
--dump_path model_cvt/ \
--from_safetensors
```


<!-- <details>
  ## ▶️ Demo
  <summary><b>Webui</b></summary>


</details> -->


<!-- <details>
  ## 👀 Visualization

  ## ⚠️ FAQ

<details> <summary><b>Question-and-Answer</b></summary>


</details> -->


## 📖 Citation
```bib
@misc{zeng2023paint3d,
      title={Paint3D: Paint Anything 3D with Lighting-Less Texture Diffusion Models},
      author={Xianfang Zeng and Xin Chen and Zhongqi Qi and Wen Liu and Zibo Zhao and Zhibin Wang and BIN FU and Yong Liu and Gang Yu},
      year={2023},
      eprint={2312.13913},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



## Acknowledgments

Thanks to [TEXTure](https://github.com/TEXTurePaper/TEXTurePaper), 
[Text2Tex](https://github.com/daveredrum/Text2Tex), 
[Stable Diffusion](https://github.com/CompVis/stable-diffusion) and [ControlNet](https://github.com/lllyasviel/ControlNet), our code is partially borrowing from them. 
Our approach is inspired by [MotionGPT](https://github.com/OpenMotionLab/MotionGPT), [Michelangelo](https://neuralcarver.github.io/michelangelo/) and [DreamFusion](https://dreamfusion3d.github.io/).

## License

This code is distributed under an [Apache 2.0 LICENSE](LICENSE).

Note that our code depends on other libraries, including [PyTorch3D](https://pytorch3d.org/) and [PyTorch Lightning](https://lightning.ai/), and uses datasets which each have their own respective licenses that must also be followed.
