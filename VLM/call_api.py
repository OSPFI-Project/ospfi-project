#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""OpenAI-compatible multimodal clients for selecting labeled image points."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from .vlm_config import VLM_API_BASE, VLM_API_KEY, VLM_MODEL
except ImportError:
    try:
        from vlm_config import VLM_API_BASE, VLM_API_KEY, VLM_MODEL
    except ImportError:
        VLM_MODEL = os.environ.get("VLM_MODEL", os.environ.get("OPENAI_VLM_MODEL", ""))
        VLM_API_KEY = os.environ.get("VLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
        VLM_API_BASE = os.environ.get("VLM_API_BASE", os.environ.get("OPENAI_BASE_URL", ""))


# ============================================================
# Configuration
# ============================================================

@dataclass
class APIConfig:
    """Runtime configuration for the multimodal clients."""

    model: str
    api_base: str
    image_detail: str = "high"
    timeout_seconds: float = 120.0
    max_retries: int = 3
    max_images: int = 20
    system_instruction: str = ""


# ============================================================
# Prompt construction
# ============================================================
def build_part_point_selection_prompt(object_name: str, part_name: str) -> str:
    """Build a prompt that selects labeled samples belonging to a target part."""

    object_name = object_name.strip()
    part_name = part_name.strip()

    if not object_name:
        raise ValueError("object_name 不能为空")

    if not part_name:
        raise ValueError("part_name 不能为空")

    prompt = f"""
        You are given two uploaded images of the same scene.

        Image 1 is a point-indexed image.
        In Image 1:
        - The whole object "{object_name}" has already been segmented by a mask.
        - A set of uniformly sampled points has been placed inside the object mask.
        - Each sampled point is marked with a unique uppercase letter.

        Image 2 is the original unmodified image of the same scene.

        Your task is to identify which points with uppercase letter belong to the target part "{part_name}" of the object "{object_name}".

        Important requirements:
        1. Select the points that clearly lie on the target part "{part_name}".
        2. If a point appears to lie on or very near the boundary/edge of the target part "{part_name}", also select it. But don't select point lie on or very near the boundary/edge of other parts!!.
        3. Do not miss any point that clearly belongs to the target part "{part_name}" or appears to be on the edge of "{part_name}".
        4. Do NOT select points that belong to other parts of the object "{object_name}" other than "{part_name}" Think twice before selecting.
        5. Use both images together:
        - Use Image 1 to read the point indices and see the masked object region.
        - Use Image 2 to understand the real object appearance and the semantic location of the target part "{part_name}".
        6. Do not include explanations, markdown, bullet points, or code fences. Return only a single valid JSON object.
        7. Usually the number of selected points is less than three, and the selected points are usually close to each other.


        Return only the uppercase letters of the selected points.

        Output format:
        {{"[letter1, letter2, letter3]}}

        If no point clearly belongs to "{part_name}" or appears to be on the edge of "{part_name}", return:
        {{"[]}}
        """.strip()

    return prompt





def build_key_interaction_point_selection_prompt(
    task_name: str,
    tool_object_name: str,
    tool_part_name: str,
    target_object_name: str,
    target_part_name: str,
) -> str:
    """
    Build a prompt that selects the tool point most critical to the interaction.

    Image 1 contains labeled projected tool points. Image 2 is the original
    interaction-start frame. Additional images show the interaction sequence.
    """

    task_name = task_name.strip()
    tool_object_name = tool_object_name.strip()
    tool_part_name = tool_part_name.strip()
    target_object_name = target_object_name.strip()
    target_part_name = target_part_name.strip()

    if not task_name:
        raise ValueError("task_name 不能为空")

    if not tool_object_name:
        raise ValueError("tool_object_name 不能为空")

    if not tool_part_name:
        raise ValueError("tool_part_name 不能为空")

    if not target_object_name:
        raise ValueError("target_object_name 不能为空")

    if not target_part_name:
        raise ValueError("target_part_name 不能为空")

    prompt = f"""
        You are given multiple uploaded images from a single robot demonstration.

        Task:
        "{task_name}"

        Objects and parts:
        - Tool object: "{tool_object_name}"
        - Tool part: "{tool_part_name}"
        - Target object: "{target_object_name}"
        - Target part: "{target_part_name}"

        Image 1 is a point-indexed image.
        In Image 1:
        - The tool part "{tool_part_name}" of the tool object "{tool_object_name}" has already been segmented or localized.
        - A set of sampled 3D template points from the tool part has been projected onto the image.
        - Each sampled point is marked with a unique uppercase letter.

        Image 2 is the original unmodified image at the interaction-start keyframe.

        Image 3 and any following images show the interaction process after the interaction-start keyframe.
        These images show how the tool part "{tool_part_name}" interacts with the target part "{target_part_name}" of the target object "{target_object_name}".

        Your task is to select exactly ONE uppercase-letter point from Image 1.

        Select the point on the tool part "{tool_part_name}" that is most critical for completing the task "{task_name}".

        The selected point should be the point that best satisfies the following criteria:
        1. It is on the tool part "{tool_part_name}".
        2. It is most tightly involved in the physical interaction with the target part "{target_part_name}".
        3. It is closest to, first approaches, or most directly acts on the target part during the interaction process.
        4. Its misalignment would most likely cause the task to fail.
        5. It best represents the functional contact or near-contact location of the tool part.
        6. If there are more than one point that seems equally critical, choose the one that is closest to the target part "{target_part_name}".

        
        Important requirements:
        1. Use Image 1 to read the uppercase-letter point labels.
        2. Use Image 2 to understand the spatial relationship at the interaction-start keyframe.
        3. Use Image 3 and the following images to understand the actual interaction process.
        4. Do not simply choose the geometric center of the tool part unless it is truly the most functionally important interaction point.
        5. Do not select a point only because it is visually prominent, large, central, or easy to see.
        6. Do not select any point that belongs to another part of the tool object "{tool_object_name}".
        7. If multiple points seem plausible, choose the single point most directly involved in the physical interaction with the target part "{target_part_name}".
        8. If no point is clearly suitable, return an empty JSON array.
        9. Do not include explanations, markdown, bullet points, or code fences. Return only a single valid JSON array.

        Return only the uppercase letter of the selected point.

        Output format:
        ["A"]

        If no point is clearly suitable, return:
        []
        """.strip()

    return prompt




# ============================================================
# Multimodal API clients
# ============================================================

class MultimodalPointSelectionClient:
    """
    Select labeled image points that belong to a specified object part.

    Example:
        client = MultimodalPointSelectionClient(...)
        result = client.select_part_points(
            object_name="wooden mug tree",
            part_name="horizontal peg",
            image_paths=[
                Path("example_fps_points_indexed.png"),
                Path("original.png")
            ],
            output_path=Path("outputs/result.json")
        )
    """

    SUPPORTED_IMAGE_SUFFIXES = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
    }

    def __init__(
        self,
        config: APIConfig | None = None,
        enable_logging: bool = True,
    ):
        if enable_logging:
            self.setup_logging()

        if OpenAI is None:
            raise RuntimeError("缺少 openai Python 包，无法调用多模态 VLM API。")

        self.validate_vlm_config()

        if config is None:
            config = APIConfig(
                model=VLM_MODEL.strip(),
                api_base=VLM_API_BASE.strip(),
            )

        self.config = config

        self.client = OpenAI(
            api_key=VLM_API_KEY,
            base_url=self.config.api_base,
            timeout=self.config.timeout_seconds,
            max_retries=0,
        )

        logging.info("模型：%s", self.config.model)
        logging.info("API Base:%s", self.config.api_base)

    # --------------------------------------------------------
    # Logging and configuration
    # --------------------------------------------------------

    @staticmethod
    def setup_logging() -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    @staticmethod
    def validate_vlm_config() -> None:
        if not isinstance(VLM_MODEL, str) or not VLM_MODEL.strip():
            raise RuntimeError("vlm_config.py 中的 VLM_MODEL 不能为空")

        if not isinstance(VLM_API_KEY, str) or not VLM_API_KEY.strip():
            raise RuntimeError("vlm_config.py 中的 VLM_API_KEY 不能为空")

        if not isinstance(VLM_API_BASE, str) or not VLM_API_BASE.strip():
            raise RuntimeError("vlm_config.py 中的 VLM_API_BASE 不能为空")

        if not VLM_API_BASE.startswith(("http://", "https://")):
            raise RuntimeError("VLM_API_BASE 必须是完整 URL")

    # --------------------------------------------------------
    # Image validation and encoding
    # --------------------------------------------------------

    def validate_image_path(self, image_path: Path) -> None:
        if not image_path.exists():
            raise FileNotFoundError(f"图像不存在：{image_path}")

        if not image_path.is_file():
            raise ValueError(f"图像路径不是文件：{image_path}")

        suffix = image_path.suffix.lower()

        if suffix not in self.SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(
                f"不支持的图像格式：{suffix}; "
                f"当前支持：{sorted(self.SUPPORTED_IMAGE_SUFFIXES)}"
            )

    @staticmethod
    def get_image_mime_type(image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(image_path.name)

        if mime_type is None or not mime_type.startswith("image/"):
            raise ValueError(f"无法识别图像 MIME 类型：{image_path}")

        return mime_type

    def image_to_data_url(self, image_path: Path) -> str:
        self.validate_image_path(image_path)

        mime_type = self.get_image_mime_type(image_path)

        with image_path.open("rb") as file:
            encoded_text = base64.b64encode(file.read()).decode("utf-8")

        return f"data:{mime_type};base64,{encoded_text}"

    # --------------------------------------------------------
    # Request construction
    # --------------------------------------------------------

    def build_multimodal_content(
        self,
        prompt: str,
        image_paths: Sequence[Path],
    ) -> list[dict[str, Any]]:
        if self.config.image_detail not in {"low", "high", "auto"}:
            raise ValueError(
                "image_detail 必须为 low、high 或 auto，"
                f"当前值为：{self.config.image_detail}"
            )

        if not image_paths:
            raise ValueError("至少需要提供一张图像")

        if len(image_paths) > self.config.max_images:
            raise ValueError(
                f"输入图片数量为 {len(image_paths)}，"
                f"超过当前脚本限制 {self.config.max_images}。"
            )

        resolved_paths = [Path(p).expanduser().resolve() for p in image_paths]

        for image_path in resolved_paths:
            self.validate_image_path(image_path)

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": prompt,
            }
        ]

        for image_index, image_path in enumerate(resolved_paths, start=1):
            data_url = self.image_to_data_url(image_path)

            content.append(
                {
                    "type": "text",
                    "text": f"Image {image_index}: {image_path.name}",
                }
            )

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                        "detail": self.config.image_detail,
                    },
                }
            )

        return content

    def build_messages(
        self,
        prompt: str,
        image_paths: Sequence[Path],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []

        if self.config.system_instruction.strip():
            messages.append(
                {
                    "role": "system",
                    "content": self.config.system_instruction.strip(),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": self.build_multimodal_content(
                    prompt=prompt,
                    image_paths=image_paths,
                ),
            }
        )

        return messages


    # --------------------------------------------------------
    # API request
    # --------------------------------------------------------
    def call_model(
        self,
        prompt: str,
        image_paths: Sequence[Path], ) -> Any:
        
        messages = self.build_messages(
            prompt=prompt,
            image_paths=image_paths,
        )

        last_exception: Exception | None = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                logging.info(
                    "正在调用模型 %s,第 %d/%d 次尝试",
                    self.config.model,
                    attempt,
                    self.config.max_retries,
                )

                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                )

                return response

            except Exception as exc:
                last_exception = exc

                logging.warning(
                    "第 %d 次调用失败：%s",
                    attempt,
                    exc,
                )

                if attempt < self.config.max_retries:
                    sleep_seconds = min(2 ** (attempt - 1), 8)
                    logging.info("%d 秒后重试", sleep_seconds)
                    time.sleep(sleep_seconds)

        raise RuntimeError(
            f"模型调用失败，已尝试 {self.config.max_retries} 次"
        ) from last_exception



    @staticmethod
    def extract_output_text(response: Any) -> str:
        choices = getattr(response, "choices", None)

        if not choices:
            return ""
        message = getattr(choices[0], "message", None)

        if message is None:
            return ""
        content = getattr(message, "content", None)

        if content is None:
            return ""

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts: list[str] = []

            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                else:
                    text = getattr(item, "text", None)

                if text:
                    text_parts.append(str(text))

            return "\n".join(text_parts).strip()

        return str(content).strip()

    @staticmethod
    def response_to_dict(response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump()

        if hasattr(response, "to_dict"):
            return response.to_dict()

        return {
            "raw_response": str(response),
        }

    @staticmethod
    def parse_selected_point_indices(output_text: str) -> list[int]:
        """
        Parse labeled point indices from model output.

        Expected output:
            {"selected_point_indices": ["J", "O", "G"]}

        Letters map to zero-based indices. Numeric entries and Markdown code
        fences are also accepted.
        """
        text = output_text.strip()

        if text.startswith("```"):
            lines = text.splitlines()
            lines = [
                line for line in lines
                if not line.strip().startswith("```")
            ]
            text = "\n".join(lines).strip()

        items = None

        try:
            data = json.loads(text)

            if isinstance(data, dict):
                items = data.get("selected_point_indices", [])
            elif isinstance(data, list):
                items = data

        except Exception:
            letters = re.findall(r"(?<![A-Za-z])[A-Z](?![A-Za-z])", text)
            if letters:
                items = letters

        if not isinstance(items, list):
            return []

        parsed = []

        for item in items:
            if isinstance(item, str):
                value = item.strip().upper()

                if len(value) == 1 and "A" <= value <= "Z":
                    parsed.append(ord(value) - ord("A"))
                    continue

                try:
                    parsed.append(int(value))
                    continue
                except Exception:
                    continue

            try:
                parsed.append(int(item))
            except Exception:
                continue

        return sorted(set(parsed))



    def save_result(
        self,
        output_path: Path,
        response: Any,
        output_text: str,
        prompt: str,
        image_paths: Sequence[Path],
        object_name: str,
        part_name: str,
        selected_point_indices: list[int],
    ) -> None:
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        safe_config = asdict(self.config)

        result = {
            "object_name": object_name,
            "part_name": part_name,
            "selected_point_indices": selected_point_indices,
            "config": safe_config,
            "prompt": prompt,
            "images": [
                str(Path(image_path).expanduser().resolve())
                for image_path in image_paths
            ],
            "output_text": output_text,
            "raw_response": self.response_to_dict(response),
        }

        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logging.info("结果已保存：%s", output_path)


    def select_part_points(
        self,
        object_name: str,
        part_name: str,
        image_paths: Sequence[Path],
        output_path: Path | None = None,) -> dict[str, Any]:
        """
        Select part points from an indexed image and its original RGB image.

        Args:
            object_name:
                Name of the complete object.

            part_name:
                Name of the target part.

            image_paths:
                Indexed-mask image followed by the original image.

            output_path:
                Optional JSON output path.

        Returns:
            result dict
        """

        if len(image_paths) != 2:
            raise ValueError(
                "该任务建议严格输入两张图片：\n"
                "1. 带 mask 和 FPS 点编号的图片\n"
                "2. 原始未处理图片"
            )

        prompt = build_part_point_selection_prompt(
            object_name=object_name,
            part_name=part_name,
        )
        print(prompt)
        logging.info("整体物体名称：%s", object_name)
        logging.info("目标部件名称：%s", part_name)
        logging.info("输入图像数量：%d", len(image_paths))

        for index, image_path in enumerate(image_paths, start=1):
            logging.info("图像 %d：%s", index, image_path)

        response = self.call_model(
            prompt=prompt,
            image_paths=image_paths,
        )

        output_text = self.extract_output_text(response)
        print("\n========== RAW RESPONSE DEBUG ==========\n")
        try:
            print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))
        except Exception:
            print(response)
        print("\n========================================\n")
        selected_point_indices = self.parse_selected_point_indices(output_text)

        result = {
            "object_name": object_name,
            "part_name": part_name,
            "prompt": prompt,
            "output_text": output_text,
            "selected_point_indices": selected_point_indices,
            "response": response,
        }

        if output_text:
            print("\n========== 模型输出 ==========\n")
            print(output_text)
            print("\n==============================\n")
        else:
            logging.warning("API 调用成功，但没有提取到文本结果")

        print("解析得到的 selected_point_indices:", selected_point_indices)

        if output_path is not None:
            self.save_result(
                output_path=output_path,
                response=response,
                output_text=output_text,
                prompt=prompt,
                image_paths=image_paths,
                object_name=object_name,
                part_name=part_name,
                selected_point_indices=selected_point_indices,
            )

        return result



class MultimodalKeyInteractionPointClient:
    """
    Select the labeled tool point most critical to the interaction.

    Example:
        client = MultimodalKeyInteractionPointClient(...)

        result = client.select_key_interaction_point(
            task_name="hang a mug handle onto a mug tree peg",
            tool_object_name="mug",
            tool_part_name="mug handle",
            target_object_name="wooden mug tree",
            target_part_name="peg",
            image_paths=[
                Path("tool_part_points_indexed.png"),
                Path("interaction_start_original.png"),
                Path("interaction_step_01.png"),
                Path("interaction_step_02.png"),
                Path("interaction_step_03.png"),
            ],
            output_path=Path("outputs/key_interaction_point.json")
        )

    The returned selected_point_index is zero-based:
        A -> 0, B -> 1, ..., Z -> 25
    """

    SUPPORTED_IMAGE_SUFFIXES = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
    }

    def __init__(
        self,
        config: APIConfig | None = None,
        enable_logging: bool = True,
    ):
        if enable_logging:
            self.setup_logging()

        if OpenAI is None:
            raise RuntimeError("缺少 openai Python 包，无法调用多模态 VLM API。")

        self.validate_vlm_config()

        if config is None:
            config = APIConfig(
                model=VLM_MODEL.strip(),
                api_base=VLM_API_BASE.strip(),
            )

        self.config = config

        self.client = OpenAI(
            api_key=VLM_API_KEY,
            base_url=self.config.api_base,
            timeout=self.config.timeout_seconds,
            max_retries=0,
        )

        logging.info("模型：%s", self.config.model)
        logging.info("API Base:%s", self.config.api_base)

    # --------------------------------------------------------
    # Logging and configuration
    # --------------------------------------------------------

    @staticmethod
    def setup_logging() -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    @staticmethod
    def validate_vlm_config() -> None:
        if not isinstance(VLM_MODEL, str) or not VLM_MODEL.strip():
            raise RuntimeError("vlm_config.py 中的 VLM_MODEL 不能为空")

        if not isinstance(VLM_API_KEY, str) or not VLM_API_KEY.strip():
            raise RuntimeError("vlm_config.py 中的 VLM_API_KEY 不能为空")

        if not isinstance(VLM_API_BASE, str) or not VLM_API_BASE.strip():
            raise RuntimeError("vlm_config.py 中的 VLM_API_BASE 不能为空")

        if not VLM_API_BASE.startswith(("http://", "https://")):
            raise RuntimeError("VLM_API_BASE 必须是完整 URL")

    # --------------------------------------------------------
    # Image validation and encoding
    # --------------------------------------------------------

    def validate_image_path(self, image_path: Path) -> None:
        if not image_path.exists():
            raise FileNotFoundError(f"图像不存在：{image_path}")

        if not image_path.is_file():
            raise ValueError(f"图像路径不是文件：{image_path}")

        suffix = image_path.suffix.lower()

        if suffix not in self.SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(
                f"不支持的图像格式：{suffix}; "
                f"当前支持：{sorted(self.SUPPORTED_IMAGE_SUFFIXES)}"
            )

    @staticmethod
    def get_image_mime_type(image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(image_path.name)

        if mime_type is None or not mime_type.startswith("image/"):
            raise ValueError(f"无法识别图像 MIME 类型：{image_path}")

        return mime_type

    def image_to_data_url(self, image_path: Path) -> str:
        self.validate_image_path(image_path)

        mime_type = self.get_image_mime_type(image_path)

        with image_path.open("rb") as file:
            encoded_text = base64.b64encode(file.read()).decode("utf-8")

        return f"data:{mime_type};base64,{encoded_text}"

    # --------------------------------------------------------
    # Request construction
    # --------------------------------------------------------

    def build_multimodal_content(
        self,
        prompt: str,
        image_paths: Sequence[Path],
    ) -> list[dict[str, Any]]:
        if self.config.image_detail not in {"low", "high", "auto"}:
            raise ValueError(
                "image_detail 必须为 low、high 或 auto,"
                f"当前值为：{self.config.image_detail}"
            )

        if not image_paths:
            raise ValueError("至少需要提供一张图像")

        if len(image_paths) < 2:
            raise ValueError(
                "该任务至少需要两张图片：\n"
                "1. 工具部件带投影点和编号的图像\n"
                "2. 交互起始关键帧原始图像\n"
                "可选:3. 后续交互过程图像"
            )

        if len(image_paths) > self.config.max_images:
            raise ValueError(
                f"输入图片数量为 {len(image_paths)},"
                f"超过当前脚本限制 {self.config.max_images}。"
            )

        resolved_paths = [Path(p).expanduser().resolve() for p in image_paths]

        for image_path in resolved_paths:
            self.validate_image_path(image_path)

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": prompt,
            }
        ]

        for image_index, image_path in enumerate(resolved_paths, start=1):
            data_url = self.image_to_data_url(image_path)

            content.append(
                {
                    "type": "text",
                    "text": f"Image {image_index}: {image_path.name}",
                }
            )

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                        "detail": self.config.image_detail,
                    },
                }
            )

        return content

    def build_messages(
        self,
        prompt: str,
        image_paths: Sequence[Path],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []

        if self.config.system_instruction.strip():
            messages.append(
                {
                    "role": "system",
                    "content": self.config.system_instruction.strip(),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": self.build_multimodal_content(
                    prompt=prompt,
                    image_paths=image_paths,
                ),
            }
        )

        return messages

    # --------------------------------------------------------
    # API request
    # --------------------------------------------------------

    def call_model(
        self,
        prompt: str,
        image_paths: Sequence[Path],
    ) -> Any:
        messages = self.build_messages(
            prompt=prompt,
            image_paths=image_paths,
        )

        last_exception: Exception | None = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                logging.info(
                    "正在调用模型 %s，第 %d/%d 次尝试",
                    self.config.model,
                    attempt,
                    self.config.max_retries,
                )

                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                )

                return response

            except Exception as exc:
                last_exception = exc

                logging.warning(
                    "第 %d 次调用失败：%s",
                    attempt,
                    exc,
                )

                if attempt < self.config.max_retries:
                    sleep_seconds = min(2 ** (attempt - 1), 8)
                    logging.info("%d 秒后重试", sleep_seconds)
                    time.sleep(sleep_seconds)

        raise RuntimeError(
            f"模型调用失败，已尝试 {self.config.max_retries} 次"
        ) from last_exception

    # --------------------------------------------------------
    # Response parsing
    # --------------------------------------------------------

    @staticmethod
    def extract_output_text(response: Any) -> str:
        choices = getattr(response, "choices", None)

        if not choices:
            return ""

        message = getattr(choices[0], "message", None)

        if message is None:
            return ""

        content = getattr(message, "content", None)

        if content is None:
            return ""

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts: list[str] = []

            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                else:
                    text = getattr(item, "text", None)

                if text:
                    text_parts.append(str(text))

            return "\n".join(text_parts).strip()

        return str(content).strip()

    @staticmethod
    def response_to_dict(response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump()

        if hasattr(response, "to_dict"):
            return response.to_dict()

        return {
            "raw_response": str(response),
        }

    @staticmethod
    def parse_selected_key_point_index(output_text: str) -> int | None:
        """
        Parse the selected interaction point k* from model output.

        Expected output:
            ["A"]

        Mapping:
            A -> 0, B -> 1, ..., Z -> 25

        An empty list returns None. Markdown code fences, dictionary-wrapped
        values, and a directly emitted uppercase label are accepted.
        """
        text = output_text.strip()

        if not text:
            return None

        if text.startswith("```"):
            lines = text.splitlines()
            lines = [
                line for line in lines
                if not line.strip().startswith("```")
            ]
            text = "\n".join(lines).strip()

        items = None

        try:
            data = json.loads(text)

            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, list):
                        items = value
                        break
                    if isinstance(value, str):
                        items = [value]
                        break

        except Exception:
            letters = re.findall(r"(?<![A-Za-z])[A-Z](?![A-Za-z])", text)
            if letters:
                items = letters

        if not isinstance(items, list) or not items:
            return None

        for item in items:
            if isinstance(item, str):
                value = item.strip().upper()

                if len(value) == 1 and "A" <= value <= "Z":
                    return ord(value) - ord("A")

                try:
                    int_value = int(value)
                    if int_value >= 0:
                        return int_value
                except Exception:
                    continue

            try:
                int_value = int(item)
                if int_value >= 0:
                    return int_value
            except Exception:
                continue

        return None

    # --------------------------------------------------------
    # Result serialization
    # --------------------------------------------------------

    def save_result(
        self,
        output_path: Path,
        response: Any,
        output_text: str,
        prompt: str,
        image_paths: Sequence[Path],
        task_name: str,
        tool_object_name: str,
        tool_part_name: str,
        target_object_name: str,
        target_part_name: str,
        selected_key_point_index: int | None,
    ) -> None:
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        safe_config = asdict(self.config)

        result = {
            "task_name": task_name,
            "tool_object_name": tool_object_name,
            "tool_part_name": tool_part_name,
            "target_object_name": target_object_name,
            "target_part_name": target_part_name,
            "selected_key_point_index": selected_key_point_index,
            "config": safe_config,
            "prompt": prompt,
            "images": [
                str(Path(image_path).expanduser().resolve())
                for image_path in image_paths
            ],
            "output_text": output_text,
            "raw_response": self.response_to_dict(response),
        }

        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logging.info("结果已保存：%s", output_path)

    # --------------------------------------------------------
    # Public interface
    # --------------------------------------------------------

    def select_key_interaction_point(
        self,
        task_name: str,
        tool_object_name: str,
        tool_part_name: str,
        target_object_name: str,
        target_part_name: str,
        image_paths: Sequence[Path],
        output_path: Path | None = None,
    ) -> dict[str, Any]:
        """
        Select the interaction-critical point k* on the tool part.

        Args:
            task_name:
                Natural-language task name.

            tool_object_name:
                Name of the complete tool object.

            tool_part_name:
                Name of the tool part.

            target_object_name:
                Name of the complete target object.

            target_part_name:
                Name of the target part.

            image_paths:
                Indexed tool-point image, interaction-start image, and optional
                interaction-sequence images.

            output_path:
                Optional JSON output path.

        Returns:
            result dict
        """

        if len(image_paths) < 2:
            raise ValueError(
                "该任务至少需要两张图片：\n"
                "1. 工具部件带投影点和大写字母编号的图像\n"
                "2. 交互起始关键帧原始图像\n"
                "建议额外提供若干交互过程图像。"
            )

        prompt = build_key_interaction_point_selection_prompt(
            task_name=task_name,
            tool_object_name=tool_object_name,
            tool_part_name=tool_part_name,
            target_object_name=target_object_name,
            target_part_name=target_part_name,
        )

        print(prompt)

        logging.info("任务名称：%s", task_name)
        logging.info("工具物体：%s", tool_object_name)
        logging.info("工具部件：%s", tool_part_name)
        logging.info("目标物体：%s", target_object_name)
        logging.info("目标部件：%s", target_part_name)
        logging.info("输入图像数量：%d", len(image_paths))

        for index, image_path in enumerate(image_paths, start=1):
            logging.info("图像 %d:%s", index, image_path)

        response = self.call_model(
            prompt=prompt,
            image_paths=image_paths,
        )

        output_text = self.extract_output_text(response)

        print("\n========== RAW RESPONSE DEBUG ==========\n")
        try:
            print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))
        except Exception:
            print(response)
        print("\n========================================\n")

        selected_key_point_index = self.parse_selected_key_point_index(output_text)

        result = {
            "task_name": task_name,
            "tool_object_name": tool_object_name,
            "tool_part_name": tool_part_name,
            "target_object_name": target_object_name,
            "target_part_name": target_part_name,
            "prompt": prompt,
            "output_text": output_text,
            "selected_key_point_index": selected_key_point_index,
            "response": response,
        }

        if output_text:
            print("\n========== 模型输出 ==========\n")
            print(output_text)
            print("\n==============================\n")
        else:
            logging.warning("API 调用成功，但没有提取到文本结果")

        print("解析得到的 selected_key_point_index:", selected_key_point_index)

        if output_path is not None:
            self.save_result(
                output_path=output_path,
                response=response,
                output_text=output_text,
                prompt=prompt,
                image_paths=image_paths,
                task_name=task_name,
                tool_object_name=tool_object_name,
                tool_part_name=tool_part_name,
                target_object_name=target_object_name,
                target_part_name=target_part_name,
                selected_key_point_index=selected_key_point_index,
            )

        return result
    


# ============================================================
# Command-line arguments
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "发送两张图像给多模态模型：\n"
            "1. 带整体 mask 和 FPS 点编号的图片\n"
            "2. 原始图像\n"
            "模型将返回属于目标部件的点编号。"
        )
    )

    parser.add_argument(
        "--object-name",
        type=str,
        required=True,
        help="整个物体名称，例如 'wooden mug tree' 或 'teapot'",
    )

    parser.add_argument(
        "--part-name",
        type=str,
        required=True,
        help="目标部件名称，例如 'horizontal peg' 或 'teapot spout'",
    )

    parser.add_argument(
        "--indexed-image",
        type=Path,
        required=True,
        help="带 mask 和 FPS 点编号的图片，例如 example_fps_points_indexed.png",
    )

    parser.add_argument(
        "--original-image",
        type=Path,
        required=True,
        help="原始未处理图片",
    )

    parser.add_argument(
        "--image-detail",
        choices=["low", "high", "auto"],
        default="high",
        help="图像解析精度，默认 high",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="单次请求超时时间，单位为秒",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="最大调用尝试次数",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="脚本允许的一次最大图片数量，默认 20",
    )

    parser.add_argument(
        "--system-instruction",
        type=str,
        default="",
        help="可选系统提示词",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/part_point_selection_result.json"),
        help="结果 JSON 保存路径",
    )

    return parser.parse_args()


# ============================================================
# Command-line entry point
# ============================================================

def main() -> int:
    args = parse_arguments()

    try:
        if args.max_retries < 1:
            raise ValueError("--max-retries 必须大于或等于 1")

        if args.max_images < 1:
            raise ValueError("--max-images 必须大于或等于 1")

        config = APIConfig(
            model=VLM_MODEL.strip(),
            api_base=VLM_API_BASE.strip(),
            image_detail=args.image_detail,
            timeout_seconds=args.timeout,
            max_retries=args.max_retries,
            max_images=args.max_images,
            system_instruction=args.system_instruction,
        )

        client = MultimodalPointSelectionClient(
            config=config,
            enable_logging=True,
        )

        image_paths = [
            args.indexed_image.expanduser().resolve(),
            args.original_image.expanduser().resolve(),
        ]

        client.select_part_points(
            object_name=args.object_name,
            part_name=args.part_name,
            image_paths=image_paths,
            output_path=args.output,
        )

        return 0

    except KeyboardInterrupt:
        logging.warning("程序被用户中断")
        return 130

    except Exception as exc:
        logging.exception("程序执行失败：%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
