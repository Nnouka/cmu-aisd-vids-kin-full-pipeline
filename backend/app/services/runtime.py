from app.services.pipeline import PipelineService


pipeline_service = PipelineService()


def preload_pipeline_models() -> None:
    pipeline_service.translation_service.preload()
    pipeline_service.tts_service.preload()


def get_model_readiness() -> dict[str, bool]:
    translation_loaded = pipeline_service.translation_service.is_loaded()
    tts_loaded = pipeline_service.tts_service.is_loaded()
    return {
        "translation_model_loaded": translation_loaded,
        "tts_model_loaded": tts_loaded,
    }