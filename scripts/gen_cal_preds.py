"""Regenerate calibration-set predictions for the saved IID and DS-B rerun models (no training)."""
import json, os, sys
import numpy as np, pandas as pd
nb_path, slug = sys.argv[1], sys.argv[2]
cells = json.load(open(nb_path))["cells"]
ns = {}
for i in (1, 3, 5):
    exec("".join(cells[i]["source"]).replace("display(", "print("), ns)
import tensorflow as tf
from tensorflow.keras.layers import Layer
Q = [0.10, 0.50, 0.90]
@tf.keras.utils.register_keras_serializable(package="fd004")
class FastTemporalAttention(Layer):
    def build(self, input_shape):
        d = input_shape[-1]
        self.W = self.add_weight(name="W", shape=(d, d), initializer="glorot_uniform")
        self.b = self.add_weight(name="b", shape=(d,), initializer="zeros")
        self.v = self.add_weight(name="v", shape=(d, 1), initializer="glorot_uniform")
        super().build(input_shape)
    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        return tf.reduce_sum(x * tf.nn.softmax(tf.tensordot(e, self.v, axes=1), axis=1), axis=1)
@tf.keras.utils.register_keras_serializable(package="fd004")
def multi_quantile_pinball_loss(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, (-1, 1)), tf.float32)
    return tf.reduce_mean(tf.add_n([tf.maximum(q * (y_true - y_pred[:, i:i+1]), (q - 1.0) * (y_true - y_pred[:, i:i+1])) for i, q in enumerate(Q)]))
OUT = ns["OUT"]
test = pd.read_csv(OUT / "test_predictions_rowlevel.csv")
for seed in ns["PROTOCOL"]["model_seeds"]:
    m = tf.keras.models.load_model(OUT / "models" / f"{slug}_seed{seed}.keras")
    pt = m.predict(ns["X_test"], batch_size=512, verbose=0)
    saved = test[test.seed == seed]
    assert np.allclose(pt[:, 1], saved.q50.values, atol=1e-3), "saved model does not reproduce saved test predictions"
    pc = m.predict(ns["X_cal"], batch_size=512, verbose=0)
    pd.DataFrame({"seed": seed, "engine": ns["id_cal"], "endpoint_regime": ns["reg_cal"], "true_RUL": ns["y_cal"],
                  "q10": pc[:, 0], "q50": pc[:, 1], "q90": pc[:, 2]}).to_csv(OUT / f"calibration_predictions_seed{seed}.csv", index=False)
    print(slug, seed, "cal", len(pc), "test reproduced")
