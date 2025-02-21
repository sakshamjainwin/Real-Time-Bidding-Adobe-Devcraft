import os
import torch
import joblib
import numpy as np
import pandas as pd
import models.opnn as OPNN
import utils.config as config
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from bidRequest import BidRequest

import warnings
warnings.simplefilter("ignore", UserWarning)

args = config.init_parser()

class Bid(object):
    def __init__(self, model_name="OPNN", latent_dims=10, feature_nums=801976, field_nums=18):
        if not torch.cuda.is_available():
            torch.Tensor.cuda = lambda self, *args, **kwargs: self
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # CTR model - OPNN initialization
        if model_name == "OPNN":
            self.model = OPNN.OuterPNN(feature_nums, field_nums, latent_dims)
        else:
            print("Model not found!")
        
        self.model = self.model.to(self.device)
        model_path = os.path.join(args.save_param_dir, model_name + ".pth")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        self.field_nums = field_nums

        # Load precomputed preprocessing info.
        prep_info = joblib.load(os.path.join(args.save_param_dir, "PREPROCESSING.pkl"))
        self.features = prep_info["features"]            # e.g., ["hour", "day_of_week", "region", "city", "ad_exchange", "ad_slot_visibility", "ad_slot_format", "advertiser_id", "ad_slot_width", "ad_slot_height", "ad_slot_floor_price", "bid_id"]
        self.categorical_cols = prep_info["categorical_cols"]  # e.g., ["bid_id", "hour", "day_of_week", "region", "city", "ad_exchange", "ad_slot_visibility", "ad_slot_format", "advertiser_id"]
        self.numeric_cols = prep_info["numeric_cols"]          # e.g., ["ad_slot_width", "ad_slot_height", "ad_slot_floor_price"]
        self.n_features = len(self.features)

        # CTR and CVR model initialization and Precompute of mappings for faster feature encoding
        self.cvr_model = joblib.load(os.path.join(args.save_param_dir, "CVR.pkl"))
        self.cvr_label_encoders = joblib.load(os.path.join(args.save_param_dir, "CVRLE.pkl"))
        self.cvr_mappings = self._build_mappings(self.cvr_label_encoders)
        
    def _build_mappings(self, encoders):
        """
        For each categorical feature, convert its LabelEncoder into a simple
        dictionary mapping from category string to integer code.
        """
        mappings = {}
        for feat in self.categorical_cols:
            le = encoders.get(feat)
            if le is not None:
                mappings[feat] = {cat: idx for idx, cat in enumerate(le.classes_)}
            else:
                mappings[feat] = {}
        return mappings

    def _extract_features(self, bidRequest: BidRequest):
        # Extract basic features from the bid request
        base_features = [
            int(bidRequest.get_ad_exchange() or 0),
            int(bidRequest.get_ad_slot_width() or 0),
            int(bidRequest.get_ad_slot_height() or 0),
            1 if bidRequest.get_ad_slot_visibility() == "SecondView" else 0,
            1 if bidRequest.get_ad_slot_format() == "Fixed" else 0,
            int(float(bidRequest.get_ad_slot_floor_price() or 0)),
            int(bidRequest.get_advertiser_id() or 0)
        ]
        # Pad the features to match the expected field size
        pad_length = self.field_nums - len(base_features)
        if pad_length > 0:
            base_features.extend([0] * pad_length)
        return torch.tensor(base_features, dtype=torch.long).unsqueeze(0).to(self.device)

    def parse_timestamp(self, ts):
        """
        Convert a timestamp in the format 'yyyyMMddHHmmssSSS' to a datetime object.
        """
        ts_str = str(ts)
        ts_fixed = ts_str + "000"  # Convert 3-digit milliseconds to 6-digit microseconds
        return datetime.strptime(ts_fixed, "%Y%m%d%H%M%S%f")

    def fast_preprocess(self, bidRequest, mappings):
        """
        Build a feature vector (NumPy array, shape=(1, n_features)) for the bidRequest.
        Uses precomputed dictionary mappings for categorical features and simple float conversion for numeric features.
        Assumes the order of features is fixed (from self.features).
        """
        vec = np.empty(self.n_features, dtype=np.float32)

        # Compute time features from the timestamp.
        try:
            dt = self.parse_timestamp(bidRequest.timestamp)
        except Exception:
            dt = datetime.now()
        hour = str(dt.hour)
        day_of_week = str(dt.weekday())  # Monday=0, Sunday=6

        # Build a dictionary of raw feature values.
        raw = {
            "hour": hour,
            "day_of_week": day_of_week,
            "region": str(bidRequest.region),
            "city": str(bidRequest.city),
            "ad_exchange": str(bidRequest.ad_exchange),
            "ad_slot_visibility": str(bidRequest.ad_slot_visibility),
            "ad_slot_format": str(bidRequest.ad_slot_format),
            "advertiser_id": str(bidRequest.advertiser_id),
            "bid_id": str(bidRequest.bid_id)
        }
        # For numeric features, try to convert to float; use 0.0 if conversion fails.
        try:
            raw["ad_slot_width"] = float(bidRequest.ad_slot_width)
        except Exception:
            raw["ad_slot_width"] = 0.0
        try:
            raw["ad_slot_height"] = float(bidRequest.ad_slot_height)
        except Exception:
            raw["ad_slot_height"] = 0.0
        try:
            raw["ad_slot_floor_price"] = float(bidRequest.ad_slot_floor_price)
        except Exception:
            raw["ad_slot_floor_price"] = 0.0

        # Fill the feature vector in the fixed order.
        for i, feat in enumerate(self.features):
            if feat in self.categorical_cols:
                # Use the precomputed mapping for the categorical feature.
                val = raw.get(feat, "")
                code = mappings.get(feat, {}).get(val, None)
                # If the category is unseen, assign a default code equal to the number of known categories.
                if code is None:
                    code = len(mappings.get(feat, {}))
                vec[i] = code
            else:
                # Numeric feature: use its value directly.
                vec[i] = raw.get(feat, 0.0)
        return vec.reshape(1, -1)

    def get_bid_price(self, bidRequest: BidRequest) -> int:
        try:
            # Get key parameters
            floor_price = float(bidRequest.get_ad_slot_floor_price() or 0)
            width = int(bidRequest.get_ad_slot_width() or 0)
            height = int(bidRequest.get_ad_slot_height() or 0)
            advertiser_id = bidRequest.get_advertiser_id()

            # Value-based advertiser multiplier based on known weights and budgets
            advertiser_weights = {
                "1458": 10.0,
                "3358": 40.0,
                "3386": 10.0,
                "3427": 10.0,
                "3476": 100.0
            }
            value_multiplier = advertiser_weights.get(advertiser_id, 1.0)

            # Area-based calculations
            typical_area = 612 * 186
            actual_area = max(width * height, 1)

            # Enhanced bid calculation for 3358
            if advertiser_id == "3358":
                visibility_multiplier = 2.0 if bidRequest.get_ad_slot_visibility() == "SecondView" else 1.2
                format_multiplier = 1.5 if bidRequest.get_ad_slot_format() == "Fixed" else 1.0
                area_multiplier = min(actual_area / typical_area * 2.0, 6.0)
                
                # CVR Implementation
                x_cvr = self.fast_preprocess(bidRequest, self.cvr_mappings)
                pred_cvr = (self.cvr_model.predict(x_cvr)[0])*2
                alpha = 0.5
                offset = 300.0
            else:
                visibility_multiplier = 1.5 if bidRequest.get_ad_slot_visibility() == "SecondView" else 1.0
                format_multiplier = 1.2 if bidRequest.get_ad_slot_format() == "Fixed" else 1.0
                area_multiplier = min(actual_area / typical_area, 4.0)
                pred_cvr = 0.0
                alpha = 0.0
                offset = 200.0

            # Calculate final bid
            base_bid = floor_price * area_multiplier * visibility_multiplier * format_multiplier
            value_adjusted_bid = (base_bid * value_multiplier * (1 + alpha * pred_cvr)) + offset

            # Add controlled random variation
            variation = 1.0 + (np.random.random() * 0.2 - 0.1)
            final_bid = value_adjusted_bid * variation

            return int(final_bid) if final_bid >= floor_price else -1

        except (ValueError, TypeError):
            return -1