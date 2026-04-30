"""Keras model builders for the freMTPL2 scaling workflow."""

from __future__ import annotations

import os
import random as _py_random

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
from keras import initializers, layers, ops, saving
import tensorflow as tf


def set_global_seed(seed: int) -> None:
    _py_random.seed(seed)
    np.random.seed(seed)
    try:
        keras.utils.set_random_seed(seed)
    except Exception:
        pass


def _bias_init_from_base_rate(base_rate: float):
    return keras.initializers.Constant(float(np.log(max(base_rate, 1e-12))))


def _optimizer(config: dict):
    return keras.optimizers.AdamW(
        learning_rate=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 0.0),
        beta_2=config.get("beta_2", 0.999),
        global_clipnorm=1.0,
    )


def _inputs_and_embeddings(
    feature_names: list[str],
    cardinalities: dict[str, int],
    embedding_dim: int,
) -> tuple[dict[str, keras.KerasTensor], list[keras.KerasTensor], keras.KerasTensor]:
    input_list = {}
    embedding_list = []
    token_list = []

    for column_name in feature_names:
        inp = layers.Input(shape=(1,), dtype="int32", name=column_name)
        emb = layers.Embedding(
            input_dim=max(int(cardinalities[column_name]), 1),
            output_dim=embedding_dim,
            name=f"{column_name}_embedding",
        )(inp)
        input_list[column_name] = inp
        embedding_list.append(layers.Flatten(name=f"{column_name}_flatten")(emb))
        token_list.append(emb)

    input_list["Exposure"] = layers.Input(shape=(1,), dtype="float32", name="Exposure")
    token_tensor = layers.Concatenate(axis=1, name="feature_tokens")(token_list)
    return input_list, embedding_list, token_tensor


@keras.saving.register_keras_serializable(package="FrmtplScaling")
class FeaturePositionEmbedding(layers.Layer):
    def build(self, input_shape):
        self.pos = self.add_weight(
            name="position_embeddings",
            shape=(1, int(input_shape[1]), int(input_shape[2])),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.pos


@keras.saving.register_keras_serializable(package="FrmtplScaling")
class MultiClsTokenLayer(layers.Layer):
    def __init__(self, n_cls: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.n_cls = int(n_cls)

    def build(self, input_shape):
        self.cls = self.add_weight(
            name="multi_cls_emb",
            shape=(1, self.n_cls, int(input_shape[-1])),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        cls_tokens = ops.broadcast_to(self.cls, (batch_size, self.n_cls, ops.shape(inputs)[-1]))
        return ops.concatenate([inputs, cls_tokens], axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_cls": self.n_cls})
        return cfg


@keras.saving.register_keras_serializable(package="FrmtplScaling")
class SwapClsTokenLayer(layers.Layer):
    def __init__(self, n_swap: int, **kwargs):
        super().__init__(**kwargs)
        self.n_swap = int(n_swap)

    def build(self, input_shape):
        self.swap = self.add_weight(
            name="swap_cls_emb",
            shape=(1, self.n_swap, int(input_shape[-1])),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        swap_tokens = ops.broadcast_to(self.swap, (batch_size, self.n_swap, ops.shape(inputs)[-1]))
        return ops.concatenate([inputs, swap_tokens], axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_swap": self.n_swap})
        return cfg


@keras.saving.register_keras_serializable(package="FrmtplScaling")
class RowSwapNoise(layers.Layer):
    """Swap feature tokens across rows during training for tabular SSL."""

    def __init__(self, swap_alpha: float = 0.10, **kwargs):
        super().__init__(**kwargs)
        self.swap_alpha = float(swap_alpha)

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        n_tokens = tf.shape(inputs)[1]
        if training is False:
            mask = tf.zeros((batch_size, n_tokens, 1), dtype=inputs.dtype)
            return inputs, mask

        donor_idx = tf.random.shuffle(tf.range(batch_size))
        donor_tokens = tf.gather(inputs, donor_idx, axis=0)
        mask = tf.cast(
            tf.random.uniform((batch_size, n_tokens, 1)) < self.swap_alpha,
            inputs.dtype,
        )
        return inputs * (1.0 - mask) + donor_tokens * mask, mask

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"swap_alpha": self.swap_alpha})
        return cfg


@keras.saving.register_keras_serializable(package="FrmtplScaling")
class SwapPredictionLoss(layers.Layer):
    """Attach binary cross-entropy loss for row-swap detection."""

    def __init__(self, loss_weight: float = 0.10, **kwargs):
        super().__init__(**kwargs)
        self.loss_weight = float(loss_weight)

    def call(self, inputs, training=False):
        logits, mask = inputs
        probs = ops.sigmoid(logits)
        bce = keras.losses.binary_crossentropy(mask, probs)
        self.add_loss(self.loss_weight * ops.mean(bce))
        return probs

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"loss_weight": self.loss_weight})
        return cfg


@keras.saving.register_keras_serializable(package="FrmtplScaling")
class TransformerBlock(layers.Layer):
    def __init__(
        self,
        n_heads: int,
        ffn_dim: int,
        dropout_rate: float = 0.025,
        ffn_dropout_rate: float = 0.015,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_heads = int(n_heads)
        self.ffn_dim = int(ffn_dim)
        self.dropout_rate = float(dropout_rate)
        self.ffn_dropout_rate = float(ffn_dropout_rate)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.ffn_dropout_rate)

    def build(self, input_shape):
        token_dim = int(input_shape[-1])
        key_dim = max(1, token_dim // self.n_heads)
        self.mha = layers.MultiHeadAttention(
            num_heads=self.n_heads,
            key_dim=key_dim,
            dropout=self.dropout_rate,
            name="mha",
        )
        self.ffn = keras.Sequential(
            [
                layers.Dense(self.ffn_dim, activation="gelu"),
                layers.Dropout(self.ffn_dropout_rate),
                layers.Dense(token_dim),
            ],
            name="ffn",
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        x_norm = self.norm1(inputs)
        attn = self.mha(x_norm, x_norm, training=training)
        x = inputs + self.dropout1(attn, training=training)
        y = self.norm2(x)
        y = self.ffn(y, training=training)
        return x + self.dropout2(y, training=training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "n_heads": self.n_heads,
                "ffn_dim": self.ffn_dim,
                "dropout_rate": self.dropout_rate,
                "ffn_dropout_rate": self.ffn_dropout_rate,
            }
        )
        return cfg


@keras.saving.register_keras_serializable(package="FrmtplScaling")
class ScaleEnsembleAdapter(layers.Layer):
    def __init__(self, k: int, initializer: str = "normal_around_one", stddev: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.k = int(k)
        self.initializer_name = initializer
        self.stddev = float(stddev)
        if initializer == "normal_around_one":
            self.initializer = keras.initializers.RandomNormal(mean=1.0, stddev=self.stddev)
        else:
            self.initializer = initializers.get(initializer)

    def build(self, input_shape):
        self.r = self.add_weight(
            name="adapter_r",
            shape=(self.k, int(input_shape[-1])),
            initializer=self.initializer,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        x = ops.expand_dims(inputs, axis=1)
        r = ops.expand_dims(self.r, axis=0)
        return x * r

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "k": self.k,
                "initializer": self.initializer_name,
                "stddev": self.stddev,
            }
        )
        return cfg


@keras.saving.register_keras_serializable(package="FrmtplScaling")
class BroadcastToK(layers.Layer):
    def __init__(self, k: int, **kwargs):
        super().__init__(**kwargs)
        self.k = int(k)

    def call(self, inputs):
        x = ops.expand_dims(inputs, axis=1)
        return ops.broadcast_to(x, (ops.shape(inputs)[0], self.k, ops.shape(inputs)[-1]))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"k": self.k})
        return cfg


@keras.saving.register_keras_serializable(package="FrmtplScaling")
class IndependentDense(layers.Layer):
    def __init__(self, k: int, units: int, kernel_initializer="glorot_uniform", bias_initializer="zeros", **kwargs):
        super().__init__(**kwargs)
        self.k = int(k)
        self.units = int(units)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        feat_dim = int(input_shape[-1])
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.k, feat_dim, self.units),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.k, self.units),
            initializer=self.bias_initializer,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        out = ops.einsum("bkd,kdu->bku", inputs, self.kernel)
        return out + ops.expand_dims(self.bias, axis=0)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "k": self.k,
                "units": self.units,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
            }
        )
        return cfg


@saving.register_keras_serializable(package="FrmtplScaling")
class TabMTrainingModel(keras.Model):
    """Expose aggregate predictions while training memberwise Poisson NLL."""

    def __init__(self, inference_model, member_model, epsilon: float = 1e-7, **kwargs):
        super().__init__(inputs=inference_model.inputs, outputs=inference_model.outputs, **kwargs)
        self.inference_model = inference_model
        self.member_model = member_model
        self.epsilon = float(epsilon)
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, inputs, training=False):
        return self.inference_model(inputs, training=training)

    def train_step(self, data):
        x, y = data
        all_vars = self.inference_model.trainable_variables + self.member_model.trainable_variables
        trainable_vars = list({id(var): var for var in all_vars}.values())
        with tf.GradientTape() as tape:
            mu_k = self.member_model(x, training=True)
            mu_k = ops.maximum(mu_k, self.epsilon)
            y_safe = ops.expand_dims(ops.maximum(y, 0.0), axis=1)
            loss = ops.mean(mu_k - y_safe * ops.log(mu_k))
            if self.losses:
                loss = loss + ops.sum(self.losses)
        gradients = tape.gradient(loss, trainable_vars)
        grads_and_vars = [
            (gradient, variable)
            for gradient, variable in zip(gradients, trainable_vars, strict=True)
            if gradient is not None
        ]
        self.optimizer.apply_gradients(grads_and_vars)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        mu_k = self.member_model(x, training=False)
        mu_k = ops.maximum(mu_k, self.epsilon)
        y_safe = ops.expand_dims(ops.maximum(y, 0.0), axis=1)
        loss = ops.mean(mu_k - y_safe * ops.log(mu_k))
        if self.losses:
            loss = loss + ops.sum(self.losses)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def get_glm(feature_names, cardinalities, base_rate, config):
    keras.backend.clear_session()
    input_list, embedding_list, _ = _inputs_and_embeddings(
        feature_names, cardinalities, config.get("embedding_dim", 1)
    )
    concatenated = layers.Concatenate(axis=1, name="concatenate_embeddings")(embedding_list)
    rate = layers.Dense(
        units=1,
        activation="exponential",
        bias_initializer=_bias_init_from_base_rate(base_rate),
        kernel_initializer=keras.initializers.Zeros(),
        name="output_dense",
    )(concatenated)
    output = layers.Multiply(name="output")([rate, input_list["Exposure"]])
    model = keras.Model(inputs=input_list, outputs=output, name="glm_model")
    model.compile(optimizer=_optimizer(config), loss="poisson", jit_compile=False)
    return model


def get_ffn(feature_names, cardinalities, base_rate, config):
    keras.backend.clear_session()
    input_list, embedding_list, _ = _inputs_and_embeddings(
        feature_names, cardinalities, config.get("embedding_dim", 8)
    )
    hidden = layers.Concatenate(axis=1, name="concatenate_embeddings")(embedding_list)
    for i, units in enumerate(config.get("dense_layers", [])):
        hidden = layers.Dense(units=units, activation="linear", name=f"hidden_dense_{i+1}")(hidden)
        hidden = layers.BatchNormalization(name=f"hidden_bn_{i+1}")(hidden)
        hidden = layers.Activation("relu", name=f"hidden_relu_{i+1}")(hidden)
        if config.get("dropout", 0.0) > 0:
            hidden = layers.Dropout(config["dropout"], name=f"hidden_dropout_{i+1}")(hidden)

    rate = layers.Dense(
        units=1,
        activation="exponential",
        bias_initializer=_bias_init_from_base_rate(base_rate),
        kernel_initializer=keras.initializers.Zeros(),
        name="output_dense",
    )(hidden)
    output = layers.Multiply(name="output")([rate, input_list["Exposure"]])
    model = keras.Model(inputs=input_list, outputs=output, name="ffn_model")
    model.compile(optimizer=_optimizer(config), loss="poisson", jit_compile=False)
    return model


def _transformer_backbone(tokens, config, training_ssl: bool = False):
    hidden = FeaturePositionEmbedding(name="positional_embedding")(tokens)
    if training_ssl:
        hidden, swap_mask = RowSwapNoise(config.get("swap_alpha", 0.10), name="row_swap_noise")(hidden)
    else:
        swap_mask = None

    n_features = int(tokens.shape[1])
    hidden = MultiClsTokenLayer(config.get("n_cls", 4), name="multi_cls_tokens")(hidden)
    if training_ssl:
        hidden = SwapClsTokenLayer(n_features, name="swap_cls_tokens")(hidden)

    for i in range(config.get("n_layers", 2)):
        hidden = TransformerBlock(
            n_heads=config.get("n_heads", 2),
            ffn_dim=config.get("ffn_dim", 64),
            dropout_rate=config.get("transformer_dropout_rate", 0.025),
            ffn_dropout_rate=config.get("ffn_dropout_rate", 0.015),
            name=f"transformer_block_{i+1}",
        )(hidden)
    return hidden, swap_mask


def get_transformer_multicls(feature_names, cardinalities, base_rate, config):
    keras.backend.clear_session()
    input_list, _, token_tensor = _inputs_and_embeddings(
        feature_names, cardinalities, config.get("embedding_dim", 24)
    )
    hidden, _ = _transformer_backbone(token_tensor, config, training_ssl=False)
    n_cls = config.get("n_cls", 4)
    cls_states = layers.Lambda(lambda x: x[:, -n_cls:, :], name="select_cls_tokens")(hidden)
    if config.get("cls_layernorm", True):
        cls_states = layers.LayerNormalization(name="cls_layernorm")(cls_states)
    flat = layers.Flatten(name="flatten_cls")(cls_states)
    flat = layers.Dropout(config.get("ffn_dropout_rate", 0.015), name="head_dropout")(flat)
    rate = layers.Dense(
        units=1,
        activation="exponential",
        bias_initializer=_bias_init_from_base_rate(base_rate),
        kernel_initializer=keras.initializers.Zeros(),
        name="output_dense",
    )(flat)
    output = layers.Multiply(name="output")([rate, input_list["Exposure"]])
    model = keras.Model(inputs=input_list, outputs=output, name="transformer_multicls")
    model.compile(optimizer=_optimizer(config), loss="poisson", jit_compile=False)
    return model


def get_transformer_multicls_ssl(feature_names, cardinalities, base_rate, config):
    keras.backend.clear_session()
    input_list, _, token_tensor = _inputs_and_embeddings(
        feature_names, cardinalities, config.get("embedding_dim", 24)
    )
    n_features = len(feature_names)
    n_cls = config.get("n_cls", 4)
    hidden, swap_mask = _transformer_backbone(token_tensor, config, training_ssl=True)

    cls_states = layers.Lambda(
        lambda x: x[:, n_features : n_features + n_cls, :],
        name="select_main_cls_tokens",
    )(hidden)
    if config.get("cls_layernorm", True):
        cls_states = layers.LayerNormalization(name="cls_layernorm")(cls_states)
    flat = layers.Flatten(name="flatten_cls")(cls_states)
    flat = layers.Dropout(config.get("ffn_dropout_rate", 0.02), name="head_dropout")(flat)
    rate = layers.Dense(
        units=1,
        activation="exponential",
        bias_initializer=_bias_init_from_base_rate(base_rate),
        kernel_initializer=keras.initializers.Zeros(),
        name="output_dense",
    )(flat)
    output = layers.Multiply(name="output")([rate, input_list["Exposure"]])

    swap_states = layers.Lambda(lambda x: x[:, -n_features:, :], name="select_swap_tokens")(hidden)
    swap_hidden = layers.Dense(config.get("swap_ffn_dim", 32), activation="gelu", name="swap_dense")(
        swap_states
    )
    swap_logits = layers.Dense(1, name="swap_logits")(swap_hidden)
    SwapPredictionLoss(config.get("swap_loss_weight", 0.10), name="swap_prediction_loss")(
        [swap_logits, swap_mask]
    )

    model = keras.Model(inputs=input_list, outputs=output, name="transformer_multicls_ssl")
    model.compile(optimizer=_optimizer(config), loss="poisson", jit_compile=False)
    return model


def get_tabm_mini(feature_names, cardinalities, base_rate, config):
    keras.backend.clear_session()
    input_list, embedding_list, _ = _inputs_and_embeddings(
        feature_names, cardinalities, config.get("embedding_dim", 16)
    )
    flat = layers.Concatenate(axis=1, name="concatenate_embeddings")(embedding_list)
    k = int(config.get("k", 8))
    hidden = ScaleEnsembleAdapter(
        k=k,
        initializer=config.get("first_adapter_init", "normal_around_one"),
        stddev=config.get("first_adapter_stddev", 0.05),
        name="tabm_mini_adapter",
    )(flat)
    for i, units in enumerate(config.get("dense_layers", [])):
        hidden = layers.Dense(units, activation=None, name=f"dense_{i+1}")(hidden)
        hidden = layers.Activation("relu", name=f"relu_{i+1}")(hidden)
        if config.get("dropout", 0.0) > 0:
            hidden = layers.Dropout(config["dropout"], name=f"dropout_{i+1}")(hidden)

    if config.get("output_kernel_init", "random_normal_small") == "random_normal_small":
        output_kernel_initializer = keras.initializers.RandomNormal(
            mean=0.0,
            stddev=config.get("output_kernel_stddev", 0.01),
        )
    else:
        output_kernel_initializer = initializers.get(config.get("output_kernel_init"))

    out_k = IndependentDense(
        k=k,
        units=1,
        kernel_initializer=output_kernel_initializer,
        bias_initializer=_bias_init_from_base_rate(base_rate),
        name="output_head_k",
    )(hidden)
    rates_k = layers.Activation("exponential", name="member_rates_exp")(out_k)
    rates_mean = layers.Lambda(lambda x: ops.mean(x, axis=1), name="mean_member_rates")(rates_k)
    final = layers.Multiply(name="output")([rates_mean, input_list["Exposure"]])
    exposure_k = BroadcastToK(k, name="broadcast_exposure")(input_list["Exposure"])
    mu_k = layers.Multiply(name="per_member_mu")([rates_k, exposure_k])

    inference_model = keras.Model(input_list, final, name="tabm_mini_infer")
    member_model = keras.Model(input_list, mu_k, name="tabm_mini_members")
    model = TabMTrainingModel(
        inference_model=inference_model,
        member_model=member_model,
        name="tabm_mini",
    )
    model.compile(optimizer=_optimizer(config), jit_compile=False)
    return model


def build_model(config_name, config, feature_names, cardinalities, base_rate):
    model_type = config["type"]
    if model_type == "glm":
        return get_glm(feature_names, cardinalities, base_rate, config)
    if model_type == "ffn":
        return get_ffn(feature_names, cardinalities, base_rate, config)
    if model_type == "transformer_multicls":
        return get_transformer_multicls(feature_names, cardinalities, base_rate, config)
    if model_type == "transformer_multicls_ssl":
        return get_transformer_multicls_ssl(feature_names, cardinalities, base_rate, config)
    if model_type == "tabm_mini":
        return get_tabm_mini(feature_names, cardinalities, base_rate, config)
    raise ValueError(f"Unknown model type for {config_name}: {model_type}")
