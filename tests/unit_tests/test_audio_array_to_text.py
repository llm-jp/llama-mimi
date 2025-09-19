from torchtitan.datasets.hf_datasets import audio_array_to_text
import torchaudio
from transformers import MimiModel, AutoFeatureExtractor

audio_tokenizer = MimiModel.from_pretrained("kyutai/mimi").to("cpu")
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
print(feature_extractor.sampling_rate)

audio_path = "assets/great_day.mp3"
waveform, sample_rate = torchaudio.load(audio_path)
# load audio array
print(waveform)
print(sample_rate)


text = audio_array_to_text(waveform[0], audio_tokenizer, feature_extractor, 4)
print(text)
assert text == "<audio><1049_0><1700_1><1626_2><546_3><127_0><243_1><783_2><546_3><1880_0><243_1><1559_2><164_3><1031_0><1056_1><1178_2><546_3><1031_0><1056_1><1178_2><164_3><1268_0><283_1><1478_2><529_3><376_0><1024_1><592_2><41_3><1010_0><1235_1><1350_2><297_3><510_0><836_1><431_2><58_3><109_0><531_1><1246_2><872_3><1670_0><1925_1><582_2><377_3><1106_0><403_1><1535_2><1817_3><966_0><771_1><2044_2><1490_3><1828_0><767_1><1183_2><313_3><1412_0><647_1><584_2><1722_3><2030_0><1008_1><749_2><916_3><966_0><721_1><377_2><1953_3><909_0><1732_1><922_2><377_3><858_0><1342_1><1852_2><882_3><1344_0><807_1><1246_2><1146_3><921_0><841_1><1831_2><1591_3><1903_0><246_1><788_2><377_3><1772_0><246_1><1000_2><1002_3><186_0><342_1><1206_2><1616_3><803_0><887_1><1047_2><97_3><1928_0><1050_1><783_2><290_3><356_0><1056_1><1178_2><546_3><356_0><1056_1><783_2><1348_3><148_0><243_1><1559_2><1348_3><1850_0><243_1><783_2><1348_3><1850_0><243_1><783_2><1348_3><384_0><1039_1><742_2><1620_3></audio>"



