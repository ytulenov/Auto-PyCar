import tensorflow as tf
import tf2onnx
model = tf.keras.models.load_model('optimized.h5')
spec = (tf.TensorSpec((None, 66, 100, 1), tf.float32, name="input"),)
output_path = "optimized.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
