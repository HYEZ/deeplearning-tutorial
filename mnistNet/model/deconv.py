import tensorflow as tf 
from keras import backend
from tensorflow.keras.layers import Layer




# Ref: https://github.com/ykamikawa/tf-keras-SegNet



class MaxPoolingWithArgmax2D(Layer):
	def __init__( 	self,
					pool_size 	= (2, 2),
					strides		= (2, 2),
					padding		= 'same',
					**kwargs):

		super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
		self.padding 	= padding
		self.pool_size 	= pool_size
		self.strides 	= strides
	
	def call(self, inputs, **kwargs):
		padding 	= self.padding
		pool_size 	= self.pool_size
		strides 	= self.strides
		
		ksize   = [1, pool_size[0], pool_size[1], 1]
		padding = padding.upper()
		strides = [1, strides[0], strides[1], 1]
		output, argmax = tf.nn.max_pool_with_argmax(	inputs,
														ksize   = ksize,
														strides = strides,
														padding = padding )

		argmax = backend.cast(argmax, backend.floatx())
		#print('MaxPoolingWithArgmax2D output:',output)
		#print('MaxPoolingWithArgmax2D argmax:',argmax)
		return [output, argmax]
	
	def compute_output_shape(self, input_shape):
		ratio = (1, 2, 2, 1)
		output_shape = [ dim // ratio[idx] if dim is not None else None
						for idx, dim in enumerate(input_shape)]
		output_shape = tuple(output_shape)
		return [output_shape, output_shape]
	
	def compute_mask(self, inputs, mask=None):
		return 2 * [None]


class MaxUnpooling2D(Layer):
	def __init__(self, size=(2, 2), **kwargs):
		super(MaxUnpooling2D, self).__init__(**kwargs)
		self.size = size
		print('size:', self.size)

	def call(self, inputs, output_shape=None):
		updates, mask = inputs[0], inputs[1]
		#print('updates:', updates)
		#print('mask:', mask)
		#print('name:', self.name)


		with backend.tf.variable_scope(self.name):
			mask 		= backend.cast(mask, 'int32')
			input_shape = backend.int_shape(updates)
			#print('input_shape:',input_shape)

			#  calculation new shape
			if output_shape is None:

				output_shape = (	input_shape[0],
									input_shape[1] * self.size[0],
									input_shape[2] * self.size[1],
									input_shape[3])

				#print('output_shape:',output_shape)
			

			# calculation indices for batch, height, width and feature maps
			one_like_mask	= backend.ones_like(mask, dtype='int32')
			batch_shape 	= backend.concatenate( [[input_shape[0]], [1], [1], [1]], axis=0)
			batch_range 	= backend.reshape(	backend.tf.range(output_shape[0], dtype='int32'), shape = batch_shape)
			b 				= one_like_mask * batch_range
			y 				= mask // (output_shape[2] * output_shape[3])
			x 				= (mask // output_shape[3]) % output_shape[2]
			feature_range 	= backend.tf.range(output_shape[3], dtype='int32')
			f 				= one_like_mask * feature_range

			#print('f:', f)

			# transpose indices & reshape update values to one dimension
			updates_size 	= backend.tf.size(updates)
			#print('updates_size:', updates_size)
			indices 		= backend.transpose(backend.reshape( backend.stack([b, y, x, f]), [4, updates_size]))

			#print('indices:', indices)
			values 			= backend.reshape(updates, [updates_size])
			#print('values:', values)
			ret 			= backend.tf.scatter_nd(indices, values, output_shape)
			#print('ret:', ret)
			return ret
		


	def compute_output_shape(self, input_shape):
		mask_shape = input_shape[1]
		return ( mask_shape[0],
				 mask_shape[1]*self.size[0],
				 mask_shape[2]*self.size[1],
				 mask_shape[3]
				 )



def unpool_with_with_argmax(pooled, ind, ksize=[1, 2, 2, 1]):
	"""
	  To unpool the tensor after  max_pool_with_argmax.
	  Argumnets:
	      pooled:    the max pooled output tensor
	      ind:       argmax indices , the second output of max_pool_with_argmax
	      ksize:     ksize should be the same as what you have used to pool
	  Returns:
	      unpooled:      the tensor after unpooling
	  Some points to keep in mind ::
	      1. In tensorflow the indices in argmax are flattened, so that a maximum value at position [b, y, x, c] becomes flattened index ((b * height + y) * width + x) * channels + c
	      2. Due to point 1, use broadcasting to appropriately place the values at their right locations ! 
	"""
	# Get the the shape of the tensor in th form of a list
	input_shape = pooled.get_shape().as_list()
	# Determine the output shape
	output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
	# Ceshape into one giant tensor for better workability
	pooled_ = tf.reshape(pooled, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]])
	# The indices in argmax are flattened, so that a maximum value at position [b, y, x, c] becomes flattened index ((b * height + y) * width + x) * channels + c
	# Create a single unit extended cuboid of length bath_size populating it with continous natural number from zero to batch_size
	batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
	b = tf.ones_like(ind) * batch_range
	b_ = tf.reshape(b, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
	ind_ = tf.reshape(ind, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
	ind_ = tf.concat([b_, ind_],1)
	ref = tf.Variable(tf.zeros([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]))
	# Update the sparse matrix with the pooled values , it is a batch wise operation
	unpooled_ = tf.scatter_nd_update(ref, ind_, pooled_)
	# Reshape the vector to get the final result 
	unpooled = tf.reshape(unpooled_, [output_shape[0], output_shape[1], output_shape[2], output_shape[3]])
	return unpooled



