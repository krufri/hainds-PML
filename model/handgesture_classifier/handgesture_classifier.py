#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class HandGestureClassifier(object):
    def __init__(
        self,
        model_path='model/handgesture_classifier/handgesture_classifier.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        #process shape
        first = landmark_list[0:42]
        second = landmark_list[42:84]   
        inputdata = np.stack((np.array(first).reshape(1, 42), np.array(second).reshape(1, 42)), axis=1)
        inputdata =inputdata.astype('float32')
        
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            inputdata)
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index
