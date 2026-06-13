import hashlib
import json
import os

import torch
import random
import numpy as np

from groot_dreams.data.dataset import WrappedLeRobotSingleDataset
from groot_dreams.groot_configs import construct_modality_config_and_transforms
from groot_dreams.data.dataset_mano import MANODataset
from groot_dreams.data.dataset_video import VideoDataset
from groot_dreams.data.dataset_mlspaces import MlSpacesVideoActionDataset


class PrecomputedTextEmbeddings:
    """Loads pre-computed text embeddings by caption hash.

    Directory layout (created by scripts/precompute_text_embeddings.py):
        embeddings/XX/HASH.pt  — (512, 100352) bf16 tensor
        index.json             — {caption_string: hash} mapping (optional, for validation)
    """

    def __init__(self, emb_dir: str):
        self.emb_dir = os.path.join(emb_dir, "embeddings")
        if not os.path.isdir(self.emb_dir):
            raise FileNotFoundError(f"Pre-computed embeddings dir not found: {self.emb_dir}")
        meta_path = os.path.join(emb_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"[PrecomputedTextEmbeddings] Loaded: {meta.get('num_unique_captions', '?')} unique, "
                  f"shape={meta.get('embedding_shape', '?')}, created={meta.get('created', '?')}")
        else:
            print(f"[PrecomputedTextEmbeddings] Using {self.emb_dir} (no metadata.json)")

    @staticmethod
    def caption_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def lookup(self, caption: str) -> torch.Tensor | None:
        """Return pre-computed embedding for a caption, or None if not found."""
        h = self.caption_hash(caption)
        path = os.path.join(self.emb_dir, h[:2], f"{h}.pt")
        if os.path.exists(path):
            return torch.load(path, map_location="cpu", weights_only=True)
        return None


def is_lerobot_dataset(dataset_path: str) -> bool:
    return "gr1" in dataset_path.lower() or "g1" in dataset_path.lower() or "yam" in dataset_path.lower() or "agibot" in dataset_path.lower() or "robocasa" in dataset_path.lower()


def is_mlspaces_dataset(dataset_path: str) -> bool:
    """Molmobot / MlSpaces format — H5 + MP4 videos, loaded via Pi0's MlSpacesDataset."""
    return "molmobot" in dataset_path.lower()


class VideoActionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        height=None,
        width=None,
        dataset_path=None,
        data_split="train",
        embodiment=None,
        agibot_pad_freq10=False,
        waist_concat=False,
        single_base_index=False,
        caption_mode=None,
        fg_stride=4,
        lm_tokenize=False,
        text_embeddings_dir=None,
        use_image_aug=False,
        video_height=None,
        video_width=None,
        robocasa_video_mode="grid",
        num_held_out_episodes=0,
        eval_split=False,
    ):
        self.dataset_path = dataset_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.height = height
        self.width = width

        # Pre-computed text embeddings (optional, backward compatible)
        self.precomputed_emb = None
        if text_embeddings_dir is not None:
            self.precomputed_emb = PrecomputedTextEmbeddings(text_embeddings_dir)

        config, train_transform, test_transform = construct_modality_config_and_transforms(
            num_frames=(num_frames + 1), embodiment=embodiment, agibot_pad_freq10=agibot_pad_freq10, waist_concat=waist_concat,
            use_image_aug=use_image_aug, video_height=video_height, video_width=video_width,
            robocasa_video_mode=robocasa_video_mode,
        )  # Add an additional prefix frame as baseline to compute delta actions
        reason1_conditioning_keys = None
        if embodiment == "robocasa" and robocasa_video_mode == "left_reason1_right_wrist":
            reason1_conditioning_keys = [
                "video.robot0_agentview_right",
                "video.robot0_eye_in_hand",
            ]
        self.lerobot_dataset = WrappedLeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=config,
            transforms=train_transform if data_split == "train" else test_transform,
            embodiment_tag=embodiment,
            data_split=data_split,
            num_frames=num_frames,
            single_base_index=single_base_index,
            caption_mode=caption_mode,
            fg_stride=fg_stride,
            lm_tokenize=lm_tokenize,
            reason1_conditioning_keys=reason1_conditioning_keys,
            num_held_out_episodes=num_held_out_episodes,
            eval_split=eval_split,
        )
        print(f"Loaded lerobot {data_split} dataset from {self.dataset_path} with {len(self)} samples.")

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True

    def __getitem__(self, data_id):
        lerobot_data = self.lerobot_dataset[data_id]

        if lerobot_data["video"].shape[1] != self.num_frames:
            print(f"Warning: Expected {self.num_frames} frames, but got {lerobot_data['video'].shape[1]} frames. Randomly sampling an item instead.")
            return self.__getitem__(random.randint(0, len(self) - 1))

        data = {
            "video": lerobot_data["video"],
            "lam_video": lerobot_data["lam_video"],
            "action": lerobot_data["action"],
            "dataset": self.lerobot_dataset.dataset_name,
            "fps": lerobot_data["fps"],
            "num_frames": lerobot_data["num_frames"],

            "__key__": lerobot_data["__key__"],
            "padding_mask": lerobot_data["padding_mask"],
            "image_size": lerobot_data["image_size"],
            "ai_caption": lerobot_data["ai_caption"],
        }
        # Include state if available (WAM uses it for conditioning)
        if "state" in lerobot_data and lerobot_data["state"] is not None:
            data["state"] = lerobot_data["state"]
        if "reason1_conditioning_frames" in lerobot_data:
            data["reason1_conditioning_frames"] = lerobot_data["reason1_conditioning_frames"]
        # Pass through language head token IDs
        for lm_key in ("lm_instruction_ids", "lm_caption_ids", "lm_caption_valid"):
            if lm_key in lerobot_data:
                data[lm_key] = lerobot_data[lm_key]
        # Load pre-computed text embedding if available
        if self.precomputed_emb is not None:
            emb = self.precomputed_emb.lookup(data["ai_caption"])
            if emb is not None:
                data["t5_text_embeddings"] = emb
        return data

    def __len__(self):
        return len(self.lerobot_dataset)


class MultiVideoActionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args=None,

        dataset_path=None,
        num_frames=81,
        time_division_factor=4,
        height=None,
        width=None,
        data_split="train",
        single_base_index=False,

        deterministic_uniform_sampling=False,
        dataset_mixing_weights=None,
        restrict_len=None,

        caption_mode=None,
        fg_stride=4,
        lm_tokenize=False,
        text_embeddings_dir=None,
        use_image_aug=False,
        video_height=None,
        video_width=None,
        robocasa_video_mode="grid",
        num_held_out_episodes=0,
        eval_split=False,
    ):
        if args is not None:
            dataset_path = args.dataset_path
            height = args.height
            width = args.width
            num_frames = args.num_frames
            single_base_index = args.single_base_index
            deterministic_uniform_sampling = args.deterministic_uniform_sampling
            dataset_mixing_weights = args.dataset_mixing_weights
            robocasa_video_mode = getattr(args, "robocasa_video_mode", robocasa_video_mode)

        self.dataset_path = dataset_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.height = height
        self.width = width
        self.deterministic_uniform_sampling = deterministic_uniform_sampling
        self.dataset_mixing_weights = dataset_mixing_weights
        self.restrict_len = restrict_len
        self.caption_mode = caption_mode
        self.fg_stride = fg_stride
        self.lm_tokenize = lm_tokenize
        self.text_embeddings_dir = text_embeddings_dir
        self.use_image_aug = use_image_aug
        self.video_height = video_height
        self.video_width = video_width
        self.robocasa_video_mode = robocasa_video_mode
        self.num_held_out_episodes = num_held_out_episodes
        self.eval_split = eval_split

        dataset_paths = []
        if isinstance(dataset_path, str):
            for p in dataset_path.split(","):
                p = p.strip()
                dataset_paths.append(p)
        else:
            dataset_paths = dataset_path

        self.datasets = []
        for path in dataset_paths:
            path = path.strip()
            if "egodex_21" in path.lower():
                self.datasets.append(MANODataset(
                    converted_root=path,
                    eval_mode=(data_split == "test"),
                    num_frames=num_frames,
                    time_division_factor=time_division_factor,
                ))
                print(f"Created MANODataset for {path}")
            elif is_mlspaces_dataset(path):
                self.datasets.append(MlSpacesVideoActionDataset(
                    dataset_path=path,
                    num_frames=num_frames,
                    video_height=self.video_height if self.video_height else 256,
                    video_width=self.video_width if self.video_width else 256,
                    use_image_aug=self.use_image_aug,
                    data_split=data_split,
                ))
                print(f"Created MlSpacesVideoActionDataset for {path}")
            elif not is_lerobot_dataset(path.lower()):
                self.datasets.append(VideoDataset(
                    video_root=path,
                    num_frames=num_frames,
                ))
                print(f"Created VideoDataset for {path}")
            else:
                if "gr1" in path.lower():
                    embodiment = "gr1"
                elif "agibot" in path.lower():
                    embodiment = "agibot"
                elif "g1" in path.lower():
                    embodiment = "g1"
                elif "yam" in path.lower():
                    embodiment = "yam"
                elif "robocasa" in path.lower():
                    embodiment = "robocasa"
                else:
                    raise ValueError(f"Cannot infer embodiment from dataset path: {path}")
                self.datasets.append(VideoActionDataset(
                    dataset_path=path,
                    data_split=data_split,
                    num_frames=num_frames,
                    time_division_factor=time_division_factor,
                    height=height,
                    width=width,
                    embodiment=embodiment,
                    agibot_pad_freq10=False,
                    waist_concat=False,
                    single_base_index=single_base_index,
                    caption_mode=self.caption_mode,
                    fg_stride=self.fg_stride,
                    lm_tokenize=self.lm_tokenize,
                    text_embeddings_dir=self.text_embeddings_dir,
                    use_image_aug=self.use_image_aug,
                    video_height=self.video_height,
                    video_width=self.video_width,
                    robocasa_video_mode=self.robocasa_video_mode,
                    num_held_out_episodes=self.num_held_out_episodes,
                    eval_split=self.eval_split,
                ))
                print(f"Created VideoActionDataset for {path}")

        if self.dataset_mixing_weights is not None:
            # self.dataset_mixing_weights = [float(w) for w in self.dataset_mixing_weights]
            # assert len(self.datasets) == len(self.dataset_mixing_weights), f"Number of datasets {len(self.datasets)} does not match number of mixing weights {len(self.dataset_mixing_weights)}."
            # # assert sum(self.dataset_mixing_weights) == 1.0, f"Dataset mixing weights should sum to 1.0, got {sum(self.dataset_mixing_weights)}."
            # self.dataset_mixing_weights = np.array(self.dataset_mixing_weights) / sum(self.dataset_mixing_weights)
            # self.dataset_shuffle_indexes = [torch.randperm(len(d)).tolist() for d in self.datasets]
            # self.dataset_iter_indexes = [0 for _ in self.datasets]
            self.dataset_mixing_weights = [float(w) for w in self.dataset_mixing_weights]
            total_prob = sum(self.dataset_mixing_weights)
            self.sample_probs = [x / total_prob for x in self.dataset_mixing_weights]
    
    def __getitem__(self, data_id):
        if self.deterministic_uniform_sampling:
            min_len = min([len(d) for d in self.datasets])
            if self.restrict_len is not None:
                assert min_len >= self.restrict_len / len(self.datasets), f"restrict_len {self.restrict_len} with {len(self.datasets)} datasets is too large for smallest dataset length {min_len}."
                min_len = self.restrict_len / len(self.datasets)
            dataset = self.datasets[int(data_id / min_len)]
            data_id = int((data_id % min_len) / min_len * len(dataset))
        elif self.dataset_mixing_weights is not None:
            # # Stochastic dataset sampling with per-dataset item iteration (loosely based on tf.data.Dataset.sample_from_datasets)
            # dataset_idx = int(np.random.choice(len(self.datasets), p=self.dataset_mixing_weights))
            # dataset = self.datasets[dataset_idx]
            # data_id = self.dataset_shuffle_indexes[dataset_idx][self.dataset_iter_indexes[dataset_idx]]
            # self.dataset_iter_indexes[dataset_idx] += 1
            # if self.dataset_iter_indexes[dataset_idx] == len(dataset):
            #     self.dataset_iter_indexes[dataset_idx] = 0
            #     self.dataset_shuffle_indexes[dataset_idx] = torch.randperm(len(dataset)).tolist()
            subset = random.choices(self.datasets, self.sample_probs)[0]
            sample_idx = random.randint(0, len(subset) - 1)
            return subset[sample_idx]
        else:
            if self.restrict_len is not None:
                full_len = sum([len(d) for d in self.datasets])
                data_id = int(data_id / self.restrict_len * full_len)
            for dataset in self.datasets:
                if data_id < len(dataset):
                    break
                data_id -= len(dataset)
        return dataset[data_id]
    
    def __len__(self):
        if self.restrict_len:
            return self.restrict_len
        elif self.deterministic_uniform_sampling:
            return min([len(d) for d in self.datasets]) * len(self.datasets)
        else:
            return sum([len(d) for d in self.datasets])
