import tensorflow as tf
from tensorflow import keras
import wandb
from datetime import datetime


@tf.function
def train_step(images, labels, model, loss_function, optimiser, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).

        predictions = model(images, training=True)
        loss = loss_function(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimiser.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def val_step(images, labels, model, loss_function, test_loss, test_accuracy):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_function(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


def train(model, optimiser, loss_function, train_ds, val_ds, epochs, validate, save_path="modesl/model_tf"):
    now = datetime.now()

    # metrics for measuring loss and accuracy
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='test_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    for epoch in range(epochs):
        # reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        # train
        for images, labels in train_ds.take(-1):
            train_step(images, labels, model, loss_function, optimiser, train_loss, train_accuracy)

        # validate
        if validate is True:
            for images, labels in val_ds.take(-1):
                val_step(images, labels, model, loss_function, val_loss, val_accuracy)

        wandb.log({
            "train_loss": train_loss.result(),
            "train_accuracy": train_accuracy.result(),
            "Epoch": epoch
        })
        if validate is True:
            wandb.log({
                "validation_loss": val_loss.result(),
                "validation_accuracy": val_accuracy.result()
                })

        if validate is False and train_accuracy.result() >= 0.95:
            print("Training accuracy is above 95%, stopping...")
            break
        if validate is True and val_accuracy.result() >= 0.90:
            print("Validation accuracy is above 95%, stopping...")
            break

    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    new_save_path = save_path + '_' + dt_string
    model.save(new_save_path)


