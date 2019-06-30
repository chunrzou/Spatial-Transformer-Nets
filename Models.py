import tensorflow as tf
import Functions

class affine_transformer_mnist_model(tf.keras.Model):

	def __init__(self,input_dims,trans_output_dims):

		'''
		Refer to https://github.com/darr/spatial_transformer_networks/blob/master/stn_model.py
		Input
		-----------
			input_dims -- the input shape for the first conv layer
			trans_output_dims -- output shape for transformer (e.g. (28,28,1))
		'''

		super(affine_transformer_mnist_model,self).__init__()
		# Localization Layers
		# ------------------------
		self.layer_local_1 = tf.keras.layers.Conv2D(filters=8,kernel_size=7,activation='linear',padding="valid",input_shape=input_dims)
		self.layer_local_2 = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)
		self.layer_local_3 = tf.keras.layers.ReLU()
		self.layer_local_4 = tf.keras.layers.Conv2D(filters=10,kernel_size=5,activation='linear')
		self.layer_local_5 = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)
		self.layer_local_6 = tf.keras.layers.ReLU()
		#self.layer_local_7 = tf.keras.layers.Conv2D(filters=12,kernel_size=3,activation='linear')
		#self.layer_local_8 = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)
		#self.layer_local_9 = tf.keras.layers.ReLU()
		self.layer_local_10 = tf.keras.layers.Flatten()

		# Regressor for the affine transformation
		# --------------------------
		#self.layer_reg_1 = tf.keras.layers.Dense(units=64,activation='relu')
		self.layer_reg_2 = tf.keras.layers.Dense(units=32,activation='relu')
		self.layer_reg_3 = tf.keras.layers.Dense(units=6,activation='linear',
		              kernel_initializer=tf.keras.initializers.Zeros(),
		              bias_initializer=tf.keras.initializers.Constant([1, 0, 0, 0, 1, 0],dtype="float32"))


		#  Spatial transformer network
		# --------------------------
		self.transformer = Functions.keras_transformer_layer_V2(output_dims=trans_output_dims)
		self.transformer_reshape = tf.keras.layers.Reshape((28,28,1))

		#  Classifier
		# --------------------------
		self.layer_class_1 = tf.keras.layers.Conv2D(filters=10,kernel_size=5,padding='valid',
		                                      activation='linear')
		self.layer_class_2 = tf.keras.layers.MaxPool2D(pool_size=2)
		self.layer_class_3 = tf.keras.layers.ReLU()
		self.layer_class_4 = tf.keras.layers.Conv2D(filters=20,kernel_size=5,padding='valid',
		                                      activation='linear')
		self.layer_class_5 = tf.keras.layers.SpatialDropout2D(0.5)
		self.layer_class_6 = tf.keras.layers.MaxPool2D(pool_size=2)
		self.layer_class_7 = tf.keras.layers.ReLU()
		self.layer_class_8 = tf.keras.layers.Flatten()
		self.layer_class_9 = tf.keras.layers.Dense(units=50,activation='relu')
		self.layer_class_10 = tf.keras.layers.Dropout(0.5)
		self.layer_class_11 = tf.keras.layers.Dense(units=10,activation="softmax")
		#self.layer_class_12 = tf.keras.layers.Activation(activation=tf.nn.log_softmax)

	def build(self,input_shape):
		super(affine_transformer_mnist_model,self).build(input_shape)

	def call(self,inputs):
		# Input layer
		t_local_1 = self.layer_local_1(inputs)
		t_local_2 = self.layer_local_2(t_local_1)
		t_local_3 = self.layer_local_3(t_local_2)
		t_local_4 = self.layer_local_4(t_local_3)
		t_local_5 = self.layer_local_5(t_local_4)
		t_local_6 = self.layer_local_6(t_local_5)
		#t_local_7 = self.layer_local_7(t_local_6)
		#t_local_8 = self.layer_local_8(t_local_7)
		#t_local_9 = self.layer_local_9(t_local_7)
		t_local_10 = self.layer_local_10(t_local_6)
		#t_reg_1 = self.layer_reg_1(t_local_10)
		t_reg_2 = self.layer_reg_2(t_local_10)
		t_reg_3 = self.layer_reg_3(t_reg_2)
		t_transformer = self.transformer([inputs,t_reg_3])
		t_transformer = self.transformer_reshape(t_transformer)

		t_class_1 = self.layer_class_1(t_transformer)
		t_class_2 = self.layer_class_2(t_class_1)
		t_class_3 = self.layer_class_3(t_class_2)
		t_class_4 = self.layer_class_4(t_class_3)
		t_class_5 = self.layer_class_5(t_class_4)
		t_class_6 = self.layer_class_6(t_class_5)
		t_class_7 = self.layer_class_7(t_class_6)
		t_class_8 = self.layer_class_8(t_class_7)
		t_class_9 = self.layer_class_9(t_class_8)
		t_class_10 = self.layer_class_10(t_class_9)
		t_class_11 = self.layer_class_11(t_class_10)
		#t_class_12 = self.layer_class_12(t_class_11)
		return t_class_11


class tps_transformer_mnist_model(tf.keras.Model):

	def __init__(self,input_dims,trans_output_dims):

		'''
		Refer to https://github.com/darr/spatial_transformer_networks/blob/master/stn_model.py
		Input
		-----------
			input_dims -- the input shape for the first conv layer
			trans_output_dims -- output shape for transformer (e.g. (28,28,1))
		'''

		super(affine_transformer_mnist_model,self).__init__()
		# Localization Layers
		# ------------------------
		self.layer_local_1 = tf.keras.layers.Conv2D(filters=8,kernel_size=7,activation='linear',padding="valid",input_shape=input_dims)
		self.layer_local_2 = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)
		self.layer_local_3 = tf.keras.layers.ReLU()
		self.layer_local_4 = tf.keras.layers.Conv2D(filters=10,kernel_size=5,activation='linear')
		self.layer_local_5 = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)
		self.layer_local_6 = tf.keras.layers.ReLU()
		#self.layer_local_7 = tf.keras.layers.Conv2D(filters=12,kernel_size=3,activation='linear')
		#self.layer_local_8 = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)
		#self.layer_local_9 = tf.keras.layers.ReLU()
		self.layer_local_10 = tf.keras.layers.Flatten()

		# Regressor for the affine transformation
		# --------------------------
		#self.layer_reg_1 = tf.keras.layers.Dense(units=64,activation='relu')
		self.layer_reg_2 = tf.keras.layers.Dense(units=32,activation='relu')
		self.layer_reg_3 = tf.keras.layers.Dense(units=6,activation='linear',
		              kernel_initializer=tf.keras.initializers.Zeros(),
		              bias_initializer=tf.keras.initializers.Constant([1, 0, 0, 0, 1, 0],dtype="float32"))


		#  Spatial transformer network
		# --------------------------
		self.transformer = Functions.keras_transformer_layer_V2(output_dims=trans_output_dims)
		self.transformer_reshape = tf.keras.layers.Reshape((28,28,1))

		#  Classifier
		# --------------------------
		self.layer_class_1 = tf.keras.layers.Conv2D(filters=10,kernel_size=5,padding='valid',
		                                      activation='linear')
		self.layer_class_2 = tf.keras.layers.MaxPool2D(pool_size=2)
		self.layer_class_3 = tf.keras.layers.ReLU()
		self.layer_class_4 = tf.keras.layers.Conv2D(filters=20,kernel_size=5,padding='valid',
		                                      activation='linear')
		self.layer_class_5 = tf.keras.layers.SpatialDropout2D(0.5)
		self.layer_class_6 = tf.keras.layers.MaxPool2D(pool_size=2)
		self.layer_class_7 = tf.keras.layers.ReLU()
		self.layer_class_8 = tf.keras.layers.Flatten()
		self.layer_class_9 = tf.keras.layers.Dense(units=50,activation='relu')
		self.layer_class_10 = tf.keras.layers.Dropout(0.5)
		self.layer_class_11 = tf.keras.layers.Dense(units=10,activation="softmax")
		#self.layer_class_12 = tf.keras.layers.Activation(activation=tf.nn.log_softmax)

	def build(self,input_shape):
		super(affine_transformer_mnist_model,self).build(input_shape)

	def call(self,inputs):
		# Input layer
		t_local_1 = self.layer_local_1(inputs)
		t_local_2 = self.layer_local_2(t_local_1)
		t_local_3 = self.layer_local_3(t_local_2)
		t_local_4 = self.layer_local_4(t_local_3)
		t_local_5 = self.layer_local_5(t_local_4)
		t_local_6 = self.layer_local_6(t_local_5)
		#t_local_7 = self.layer_local_7(t_local_6)
		#t_local_8 = self.layer_local_8(t_local_7)
		#t_local_9 = self.layer_local_9(t_local_7)
		t_local_10 = self.layer_local_10(t_local_6)
		#t_reg_1 = self.layer_reg_1(t_local_10)
		t_reg_2 = self.layer_reg_2(t_local_10)
		t_reg_3 = self.layer_reg_3(t_reg_2)
		t_transformer = self.transformer([inputs,t_reg_3])
		t_transformer = self.transformer_reshape(t_transformer)

		t_class_1 = self.layer_class_1(t_transformer)
		t_class_2 = self.layer_class_2(t_class_1)
		t_class_3 = self.layer_class_3(t_class_2)
		t_class_4 = self.layer_class_4(t_class_3)
		t_class_5 = self.layer_class_5(t_class_4)
		t_class_6 = self.layer_class_6(t_class_5)
		t_class_7 = self.layer_class_7(t_class_6)
		t_class_8 = self.layer_class_8(t_class_7)
		t_class_9 = self.layer_class_9(t_class_8)
		t_class_10 = self.layer_class_10(t_class_9)
		t_class_11 = self.layer_class_11(t_class_10)
		#t_class_12 = self.layer_class_12(t_class_11)
		return t_class_11

class affine_transformer_mnist_model_legacy01(tf.keras.Model):
	def __init__(self,input_dims,output_dims):
		'''
		Refer to https://github.com/zsdonghao/Spatial-Transformer-Nets
		'''
		# Input layer
		#layer_nin = tf.keras.layers.Input(shape=input_dims,name='input_layer')
		assertIsNotNone(input_dims)
		assertIsNotNone(output_dims)
		# Localization Layers
		#layer_local_1 = tf.keras.layers.Flatten(name='layer_local_1')
		#layer_local_2 = tf.keras.layers.Dense(units=20,activation='tanh',name='layer_local_2')
		#layer_local_3 = tf.keras.layers.Dropout(rate=0.1,name='layer_local_3')
		layer_local_1 = tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=(2,2),
				activation='relu',padding="same",name="layer_local_1",input_shape=input_dims)
		layer_local_2 = tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),strides=(2,2),activation='relu',padding="same",name="layer_local_2")
		layer_local_3 = tf.keras.layers.Flatten(name="layer_local_3")
		layer_local_4 = tf.keras.layers.Dropout(0.1,name="layer_local_4")
		layer_theta = tf.keras.layers.Dense(units=6,activation='linear',name="layer_theta",
		               kernel_initializer=tf.keras.initializers.Zeros(),bias_initializer=theta_init)
		# Spatial transformer
		transformer = Functions.keras_tranformer_layer(output_dims=output_dims,name='transformer')
		# Classifier
		layer_class_1 = tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=(2,2),padding="same",name="layer_class_1",activation='relu')
		layer_class_2 = tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=(2,2),padding="same",name="layer_class_2",activation='relu')
		layer_class_3 = tf.keras.layers.Flatten(name='layer_class_3')
		layer_class_4 = tf.keras.layers.Dense(units=400,name="layer_class_4",activation="relu")
		layer_class_5 = tf.keras.layers.Dense(units=10,activation="linear",name="layer_class_5")

	def call(self,inputs):
		## Model 
		## -----------------
		t_local_1 = layer_local_1(inputs)
		t_local_2 = layer_local_2(t_local_1)
		t_local_3 = layer_local_3(t_local_2)
		t_local_4 = layer_local_4(t_local_3)
		t_theta = layer_theta(t_local_4)
		t_transformer = transformer([layer_nin,t_theta])
		t_class_1 = layer_class_1(t_transformer)
		t_class_2 = layer_class_2(t_class_1)
		t_class_3 = layer_class_3(t_class_2)
		t_class_4 = layer_class_4(t_class_3)
		t_class_5 = layer_class_5(t_class_4)

		return t_class_5


class tps_classifier(tf.keras.Model):

	def __init__(self,input_dims,output_classes,**kwargs):
		
		super(classifier,self).__init__(**kwargs)
		assert tf.rank(input_dims) == 3
		self.input_dims = input_dims
		self.layer1 = tf.keras.layers.Conv2D(filters=10,kernel_size=5,input_shape=input_dims)
		self.layer2 = tf.keras.layers.MaxPool2D(pool_size=2)
		self.layer3 = tf.keras.layers.Activation.ReLU()
		self.layer4 = tf.keras.layers.Conv2D(filters=20,kernel_size=5)
		self.layer5 = tf.keras.layers.SpatialDropout2D(rate=0.5)
		self.layer6 = tf.keras.layers.MaxPool2D(pool_size=2)
		self.layer7 = tf.keras.layers.Activation.ReLU()
		self.layer8 = tf.keras.layers.Flatten()
		self.layer9 = tf.keras.layers.Dense(units=50,activation='relu')
		self.layer10 = tf.keras.layers.Dropout(rate=0.5)
		self.layer11 = tf.keras.layers.Dense(output_classes,activation='softmax')

	def call(self,inputs):

		t_1 = self.layer1(inputs)
		t_2 = self.layer2(t_1)
		t_3 = self.layer3(t_2)
		t_4 = self.layer4(t_3)
		t_5 = self.layer5(t_4)
		t_6 = self.layer6(t_5)
		t_7 = self.layer7(t_6)
		t_8 = self.layer8(t_7)
		t_9 = self.layer9(t_8)
		t_10 = self.layer10(t_9)
		t_11 = self.layer11(t_10)

		return t_11
		
class tps_localizer(tf.keras.Model):

	def __init__(self,grid_height,grid_width,target_control_points,input_shape,bounded=True,**kwargs):
		'''
		tps_localizer will generate the source_control_point in the input images.

		Input
		--------
		grid_height -- The y dimension of the target_control_points
		grid_width -- The x dimension of the target_control_points
		target_control_points -- [x,y] of shape (N,2)
		input_shape -- The image 2D size of shape (H,W)
		bounded  -- If the grid extent is bounded from -1 to 1 or not
		'''
		super(tps_localizer,self).__init__(**kwargs)

		assert tf.shape(target_control_points)[0] == grid_width*grid_height
		self.output_dim = tf.shape(target_control_points)[0]


		self.layer1 = tf.keras.layers.Conv2D(filters=10,kernel_size=5,input_shape=input_dims)
		self.layer2 = tf.keras.layers.MaxPool2D(pool_size=2)
		self.layer3 = tf.keras.layers.Activation.ReLU()
		self.layer4 = tf.keras.layers.Conv2D(filters=20,kernel_size=5)
		self.layer5 = tf.keras.layers.SpatialDropout2D(rate=0.5)
		self.layer6 = tf.keras.layers.MaxPool2D(pool_size=2)
		self.layer7 = tf.keras.layers.Activation.ReLU()
		self.layer8 = tf.keras.layers.Flatten()
		self.layer9 = tf.keras.layers.Dense(units=50,activation='relu')
		self.layer10 = tf.keras.layers.Dropout(rate=0.5)

		if bounded:
			self.layer11 = tf.keras.layers.Dense(units=self.output_dim,activation='tanh',
				kernel_initializer=tf.keras.initializers.Zeros(),
		              bias_initializer=tf.keras.initializers.Constant(tf.atanh(target_control_points),dtype="float32"))
		else:
			self.layer11 = tf.keras.layers.Dense(units=self.output_dim,activation='linear',
				kernel_initializer=tf.keras.initializers.Zeros(),
		              bias_initializer=tf.keras.initializers.Constant(target_control_points,dtype="float32"))


	def call(self,inputs):
		t_1 = self.layer1(inputs)
		t_2 = self.layer2(t_1)
		t_3 = self.layer3(t_2)
		t_4 = self.layer4(t_3)
		t_5 = self.layer5(t_4)
		t_6 = self.layer6(t_5)
		t_7 = self.layer7(t_6)
		t_8 = self.layer8(t_7)
		t_9 = self.layer9(t_8)
		t_10 = self.layer10(t_9)
		t_11 = self.layer11(t_10)

		return t_11






