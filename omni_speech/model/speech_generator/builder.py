from .speech_generator import SpeechGeneratorCTC


def build_speech_generator(config):
    generator_type = getattr(config, 'speech_generator_type', 'ctc')
    if generator_type == 'ctc':
        return SpeechGeneratorCTC(config)

    raise ValueError(f'Unknown generator type: {generator_type}')
