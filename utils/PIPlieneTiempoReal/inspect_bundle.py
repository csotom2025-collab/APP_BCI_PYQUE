import joblib, numpy as np
import config
from eeg_features import EEGFeatureExtractor

bundle = joblib.load(r'd:\EEG_Python\PIPlieneTiempoReal\OutputOPt5\models\UserCMSM\hierarchical_bundle.joblib')
print('bundle channel_names', bundle['channel_names'])
print('first super columns', bundle['super_clase']['feature_columns'][:10])
print('count super', len(bundle['super_clase']['feature_columns']))

rng = np.random.default_rng(0)
sig = rng.normal(0,1,size=(len(config.CHANNEL_NAMES),256))
feat = EEGFeatureExtractor(fs=config.FS).extract_features(
    sig,
    channel_names=config.CHANNEL_NAMES,
    available_channel_names=config.CHANNEL_NAMES,
    window_size=sig.shape[1],
    overlap=0.0,
)
print('len extracted', len(feat.columns))
print('first extracted', list(feat.columns[:10]))
print('last extracted', list(feat.columns[-10:]))
missing = [c for c in bundle['super_clase']['feature_columns'] if c not in feat.columns]
print('missing_count', len(missing))
print('missing_first', missing[:20])
