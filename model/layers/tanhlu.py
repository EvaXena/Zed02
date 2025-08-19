import tensorflow as tf
from tensorflow.keras import layers, initializers, models

# =================================================================
#           THE CUSTOM TanhLU LAYER IMPLEMENTATION
# =================================================================

class Tanhlu(layers.Layer):
    """
    Trainable Tanh-Linear Unit Activation Layer.

    Implements the function: f(x) = alpha * tanh(lambda * x) + beta * x
    where alpha, beta, and lambda are trainable parameters.
    """
    def __init__(self,
                  alpha_initializer='ones',
                  beta_initializer='zeros',##init参数作用
                  lambd_initializer='ones',
                  **kwargs
                  ):
        super(Tanhlu,self).__init__(**kwargs)##super作用

        self.alpha_initializer = initializers.get(alpha_initializer)
        self.beta_initializer = initializers.get(beta_initializer)
        self.lambd_initializer = initializers.get(lambd_initializer)

    def build(self,input_shape):
         #创建可训练参数
        self.alpha = self.add_weight(
             shape=(1,),#这个shape的表示形状是什么意思
             name = 'alpha',
             initializer = self.alpha_initializer,
             trainable=True
            )

        self.beta = self.add_weight(
             shape=(1,),
             name = 'beta',
             initializer = self.beta_initializer,
             trainable=True
        )

        self.lambd = self.add_weight(
            shape = (1,),
            name = 'lambd',
            initializer = self.lambd_initializer,
            trainable = True
        )

        super(Tanhlu,self).build(input_shape)#这个super作用

    def get_config(self):
        config = {
            'alpha_initializer':initializers.serialize(self.alpha_initializer),
            'beta_initializer':initializers.serialize(self.beta_initializer),
            'lambd_initializer':initializers.serialize(self.lambd_initializer),
        }

        base_config = super(Tanhlu,self).get_config()#作用
        return dict(list(base_config.items()) + list(config.items()))#作用
    
    def call(self,inputs):
        return self.alpha * tf.math.tanh(self.lambd * inputs) + self.beta * inputs