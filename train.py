import tensorflow as tf


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


def train(model, optimiser, loss_function, train_ds, val_ds, epochs, validate):
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

        # print the metrics at the end of each epoch
        print(
            f'Epoch {epoch + 1},\t\t'
            f'Loss: {train_loss.result():.2f},\t'
            f'Accuracy: {train_accuracy.result() * 100:.2f},',
            end='\t\t'
            )
        if validate is True:
            print(
                f'Validation Loss: {val_loss.result():.2f},\t'
                f'Validation Accuracy: {val_accuracy.result() * 100:.2f}'
                )
