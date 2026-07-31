"""Extract text and normalized prices from price-tag image crops.

The PaddleOCR adapter is loaded lazily and shared across tag crops. Each crop is
optionally enlarged, converted into normalized price candidates, and retried
with focused or contrast-enhanced images when the first OCR pass finds no
price. Failures use the same result schema as successful extraction.
"""

from __future__ import annotations

import os
from typing import Optional

from ..image_processing.image_io import import_cv2
from .price_parsing import (
    DEFAULT_OCR_MAX_PRICES,
    DEFAULT_OCR_MIN_CONFIDENCE,
    flatten_ocr_result,
    price_candidates_from_ocr,
    select_price_candidates,
)


class PaddlePriceExtractor:
    """Extract up to three spatially ordered prices from a tag crop."""

    def __init__(
        self,
        ocr_version: str = "PP-OCRv6",
        language: str = "pt",
        min_confidence: float = DEFAULT_OCR_MIN_CONFIDENCE,
        max_prices: int = DEFAULT_OCR_MAX_PRICES,
        upscale: float = 3.0,
    ):
        """Validate and store text-extraction settings.

        Input:
            ocr_version: PaddleOCR model family to load.
            language: PaddleOCR language code for recognition.
            min_confidence: Minimum OCR confidence accepted for price parsing.
            max_prices: Maximum number of ordered prices returned per tag.
            upscale: Resize factor applied before OCR inference.
        Output:
            None. The extractor remains unloaded until its first request.
        """
        if not 1 <= int(max_prices) <= 3:
            raise ValueError("max_prices must be between 1 and 3.")
        if not 0.0 <= float(min_confidence) <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1.")
        if not 1.0 <= float(upscale) <= 4.0:
            raise ValueError("upscale must be between 1 and 4.")
        self.ocr_version = str(ocr_version)
        self.language = str(language)
        self.min_confidence = float(min_confidence)
        self.max_prices = int(max_prices)
        self.upscale = float(upscale)
        self._ocr = None
        self._load_error: Optional[str] = None

    def _load(self):
        """Load and cache the configured PaddleOCR engine.

        Input:
            None. Model settings come from extractor initialization.
        Output:
            The cached PaddleOCR engine.
        """
        if self._ocr is not None:
            return self._ocr
        if self._load_error:
            raise RuntimeError(self._load_error)
        os.environ.setdefault(
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK",
            "True",
        )
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as exc:
            self._load_error = "paddleocr_not_installed"
            raise ImportError(
                "PaddleOCR is required for price-number extraction."
            ) from exc

        try:
            self._ocr = PaddleOCR(
                lang=self.language,
                ocr_version=self.ocr_version,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except Exception as exc:
            self._load_error = (
                f"Could not initialize PaddleOCR {self.ocr_version} "
                f"for language {self.language}: {exc}"
            )
            raise RuntimeError(self._load_error) from exc
        return self._ocr

    @staticmethod
    def _empty_result(error: str) -> dict:
        """Build an empty result that preserves the extraction schema.

        Input:
            error: Machine-readable or diagnostic extraction error.
        Output:
            A price result dictionary with empty values and the supplied error.
        """
        return {
            "primary_price": None,
            "primary_value": None,
            "primary_price_confidence": None,
            "secondary_price": None,
            "secondary_value": None,
            "secondary_price_confidence": None,
            "tertiary_price": None,
            "tertiary_value": None,
            "tertiary_price_confidence": None,
            "ocr_error": error,
            "ocr_text": "",
        }

    def extract(self, crop_bgr) -> dict:
        """Extract ordered prices and audit text from one price-tag crop.

        Input:
            crop_bgr: Price-tag image crop in BGR array format.
        Output:
            A dictionary containing up to three prices, confidences, OCR text,
            and an optional extraction error.
        """
        # Model initialization failures are returned in the normal audit schema.
        try:
            ocr = self._load()
        except Exception as exc:
            return self._empty_result(str(exc))

        try:
            # Enlarge small tag crops before the first text-recognition pass.
            ocr_image = crop_bgr
            if self.upscale > 1.0 and crop_bgr is not None:
                cv2 = import_cv2()
                ocr_image = cv2.resize(
                    crop_bgr,
                    None,
                    fx=self.upscale,
                    fy=self.upscale,
                    interpolation=cv2.INTER_CUBIC,
                )
            # Normalize PaddleOCR output and select non-overlapping price values.
            raw = (
                ocr.predict(ocr_image)
                if hasattr(ocr, "predict")
                else ocr.ocr(ocr_image, cls=True)
            )
            items = flatten_ocr_result(raw)
            candidates = price_candidates_from_ocr(
                items,
                min_confidence=self.min_confidence,
            )
            selected = select_price_candidates(
                candidates,
                max_prices=self.max_prices,
            )
            # Retry difficult tags with a price-focused crop and enhanced contrast.
            if not selected and ocr_image is not None:
                cv2 = import_cv2()
                image_height = ocr_image.shape[0]
                gray = cv2.cvtColor(ocr_image, cv2.COLOR_BGR2GRAY)
                contrast = cv2.createCLAHE(
                    clipLimit=2.0,
                    tileGridSize=(8, 8),
                ).apply(gray)
                contrast_image = cv2.cvtColor(
                    contrast,
                    cv2.COLOR_GRAY2BGR,
                )
                retry_images = [
                    ocr_image[int(round(0.20 * image_height)) :],
                    contrast_image,
                ]
                for retry_image in retry_images:
                    if retry_image.size == 0:
                        continue
                    retry_raw = (
                        ocr.predict(retry_image)
                        if hasattr(ocr, "predict")
                        else ocr.ocr(retry_image, cls=True)
                    )
                    retry_items = flatten_ocr_result(retry_raw)
                    retry_candidates = price_candidates_from_ocr(
                        retry_items,
                        min_confidence=self.min_confidence,
                    )
                    retry_selected = select_price_candidates(
                        retry_candidates,
                        max_prices=self.max_prices,
                    )
                    if retry_selected:
                        items = retry_items
                        selected = retry_selected
                        break
        except Exception as exc:
            return self._empty_result(str(exc))

        # Preserve recognized audit text even when no valid price is available.
        if not selected:
            result = self._empty_result("price_not_found")
            result["ocr_text"] = " | ".join(item.text for item in items)
            return result

        # Map the spatially ordered candidates into the stable output schema.
        primary = selected[0]
        secondary = selected[1] if len(selected) > 1 else None
        tertiary = selected[2] if len(selected) > 2 else None
        return {
            "primary_price": primary.price,
            "primary_value": primary.value,
            "primary_price_confidence": primary.confidence,
            "secondary_price": secondary.price if secondary else None,
            "secondary_value": secondary.value if secondary else None,
            "secondary_price_confidence": (
                secondary.confidence if secondary else None
            ),
            "tertiary_price": tertiary.price if tertiary else None,
            "tertiary_value": tertiary.value if tertiary else None,
            "tertiary_price_confidence": (
                tertiary.confidence if tertiary else None
            ),
            "ocr_error": None,
            "ocr_text": " | ".join(item.text for item in items),
        }
