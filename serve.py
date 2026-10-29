"""Run the optimized single-thread TTS server: python serve.py --model <package>."""
import argparse
from server import DEFAULT_MODEL, create_app
import uvicorn


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL, help="Prepared F1200 or latent-retime model directory")
    p.add_argument("--model-id", default="supertonic-3", help="API model identifier")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7788)
    p.add_argument("--cors", default="", help="Comma-separated browser origins")
    p.add_argument("--max-pending", type=int, default=4)
    p.add_argument("--max-text-length", type=int, default=10000)
    p.add_argument("--max-audio-seconds", type=float, default=180)
    p.add_argument("--packet-ms", type=int, default=40)
    p.add_argument("--custom-styles-dir", default=".private/server-styles")
    a = p.parse_args()
    app = create_app(model_path=a.model, model_id=a.model_id, cors_origins=a.cors.split(",") if a.cors else None,
                     max_pending=a.max_pending, max_text_length=a.max_text_length,
                     max_audio_seconds=a.max_audio_seconds, packet_ms=a.packet_ms,
                     custom_styles_dir=a.custom_styles_dir)
    uvicorn.run(app, host=a.host, port=a.port, workers=1)


if __name__ == "__main__":
    main()
