import tensorflow as tf

class JITModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Tracker explícito de loss (promedio por epoch en logs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @tf.function(jit_compile=True, reduce_retracing=True)
    def train_step(self, data):
        x, y, sw = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # compute_loss moderno; ya incluye regularization losses
            loss = self.compute_loss(
                x=x, y=y, y_pred=y_pred, sample_weight=sw, training=True
            )

            # Si el optimizer está envuelto con LossScaleOptimizer:
            if hasattr(self.optimizer, "get_scaled_loss"):
                loss_for_grad = self.optimizer.get_scaled_loss(loss)
            else:
                loss_for_grad = loss

        grads = tape.gradient(loss_for_grad, self.trainable_variables)
        if hasattr(self.optimizer, "get_unscaled_gradients"):
            grads = self.optimizer.get_unscaled_gradients(grads)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Métricas sin compiled_metrics (API moderna)
        try:
            logs = self.compute_metrics(x=x, y=y, y_pred=y_pred, sample_weight=sw)
        except Exception:
            for m in self.metrics:
                try:
                    m.update_state(y, y_pred, sample_weight=sw)
                except TypeError:
                    m.update_state(y, y_pred)
            logs = {m.name: m.result() for m in self.metrics}

        # Actualiza y devuelve loss promedio
        self.loss_tracker.update_state(loss)
        logs = {"loss": self.loss_tracker.result(), **logs}
        return logs

    @tf.function(jit_compile=True, reduce_retracing=True)
    def test_step(self, data):
        x, y, sw = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_pred = self(x, training=False)
        loss = self.compute_loss(
            x=x, y=y, y_pred=y_pred, sample_weight=sw, training=False
        )
        try:
            logs = self.compute_metrics(x=x, y=y, y_pred=y_pred, sample_weight=sw)
        except Exception:
            for m in self.metrics:
                try:
                    m.update_state(y, y_pred, sample_weight=sw)
                except TypeError:
                    m.update_state(y, y_pred)
            logs = {m.name: m.result() for m in self.metrics}

        self.loss_tracker.update_state(loss)
        logs = {"loss": self.loss_tracker.result(), **logs}
        return logs
