# Installation

## 1. Create the Python environment

```bash
conda create -n ospfi python=3.10 -y
conda activate ospfi
python -m pip install --upgrade pip setuptools wheel
```

## 2. Install PyTorch

Install `torch` and `torchvision` with the command matching the local CUDA
version from the [official PyTorch installation page](https://pytorch.org/get-started/locally/).

Verify the installation:

```bash
python -c "import torch, torchvision; print(torch.__version__); print(torch.cuda.is_available())"
```

## 3. Install the project Python dependencies

```bash
cd OSPFI
python -m pip install -r requirements.txt
```

If `pytorch3d` cannot be resolved for the installed PyTorch and CUDA versions,
install it by following the
[official PyTorch3D installation instructions](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

## 4. Install GroundingDINO, Segment Anything, and DINOv2

Install the three visual perception modules by following the instructions in
their official repositories:

- [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
- [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)

Clone and install the GroundingDINO and Segment Anything source repositories under `GroundedSam/`. Install DINOv2 separately under `dinov2_local/`:

```bash
mkdir -p GroundedSam weight
git clone https://github.com/IDEA-Research/GroundingDINO.git GroundedSam/GroundingDINO
python -m pip install -e GroundedSam/GroundingDINO
git clone https://github.com/facebookresearch/segment-anything.git GroundedSam/segment-anything
python -m pip install -e GroundedSam/segment-anything
git clone https://github.com/facebookresearch/dinov2.git dinov2_local
python -m pip install -e dinov2_local
```

Cloning the repositories installs their source code only. Download the GroundingDINO Swin-T and SAM ViT-B checkpoints separately into the project-root `weight/` directory:

```bash
wget -O weight/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -O weight/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

The resulting source and checkpoint layout must be:

```text
GroundedSam/
├── GroundingDINO/
│   └── groundingdino/config/GroundingDINO_SwinT_OGC.py
└── segment-anything/
weight/
├── groundingdino_swint_ogc.pth
└── sam_vit_b_01ec64.pth
```

## 5. Configure the VLM API

Edit:

```text
VLM/vlm_config.py
```

Set the OpenAI-compatible model, API key, and API base:

```python
VLM_MODEL = "your-model-name"
VLM_API_KEY = "your-api-key"
VLM_API_BASE = "your-api-base"
```

## 6. Run the demo

Prepare the task files with the following layout:

```text
inference/source/<task_name>/scene_manip_rgb.png
inference/source/<task_name>/scene_manip_depth.png
template/<task_name>/manipulation_template.npz
camera/camera_parameters.json
```

Run manipulation inference from the project root:

```bash
python scripts/demo.py sweeping
```

Replace `sweeping` with another prepared task name when needed:

```bash
python scripts/demo.py hanging
python scripts/demo.py pouring
```

The inference result is saved under:

```text
inference/result/<task_name>/
```

Visualize the template keypoints and tool-center trajectory:

```bash
python scripts/visualize_template.py sweeping
```


Visualize the saved inference result:

```bash
python scripts/visualize_result.py sweeping
```
