#!/usr/bin/env python3
"""Compatibility entry point: this exports synthetic JSON, not LiteRT models."""

if __package__:
    from .export_synthetic_model import build_sample_model, export_model, main, requantize_reference
else:
    from export_synthetic_model import build_sample_model, export_model, main, requantize_reference

if __name__ == "__main__":
    main()
