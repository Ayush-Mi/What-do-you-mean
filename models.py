import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_text as text

class quantize_model:
    def __init__(self,model,custom_object=None):
        self.model = model
        self.custom_object = custom_object
        self.load_model()
    
    def dynamic_range_quantization(self,save_model=True):
        save_path = 'bert_dynamic_range.tflite'

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter._experimental_lower_tensor_list_ops = False

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_model = converter.convert()

        if save_model:
            # Save the model.
            with open(save_path, 'wb') as f:
                f.write(quantized_model)
        print("Model sucessfully quantized and saved at ",save_path)
        return quantized_model

    def float16_quantization(self,save_model=True):
        save_path = 'bert_float16.tflite'

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter._experimental_lower_tensor_list_ops = False

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16] 
        quantized_model = converter.convert()

        if save_model:
            # Save the model.
            with open(save_path, 'wb') as f:
                f.write(quantized_model)
        print("Model sucessfully quantized and saved at ",save_path)
        return quantized_model
    
    def int8_quantization(self,rep_data_path,save_model=True):
        save_path = 'bert_int8.tflite'
        self.rep_data_path = rep_data_path

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.representative_dataset = self.representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter._experimental_lower_tensor_list_ops = False

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8] 
        converter.allow_custom_ops=True
        quantized_model = converter.convert()

        if save_model:
            # Save the model.
            with open(save_path, 'wb') as f:
                f.write(quantized_model)
        print("Model sucessfully quantized and saved at ",save_path)
        return quantized_model

    def load_model(self,):
        if self.custom_object:
            self.model = tf.keras.models.load_model(self.model,custom_objects=self.custom_object)
        else:
            self.model = tf.keras.models.load_model(self.model)

    def representative_data_gen(self,):  
        with open(self.rep_data_path) as file:
            for i in file:
                yield[i]

# Custom Funciton used while training the intent deteciton model
def recall(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall 

def precision(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+K.epsilon())
        return precision 

def f1_score(y_true, y_pred):    
    
    p_recision = precision(y_true, y_pred), 
    r_call= recall(y_true, y_pred)
    
    return 2*((p_recision*r_call)/(p_recision+r_call+K.epsilon()))

if __name__ == '__main__':
    q_model = quantize_model(model='./bert_model/',custom_object={'f1_score':f1_score,'precision':precision,'recall':recall})
    #q_model.dynamic_range_quantization()
    #q_model.float16_quantization()
    q_model.int8_quantization(rep_data_path='Datasets/SNIPS/valid/seq.in')