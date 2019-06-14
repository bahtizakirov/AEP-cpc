
import torch   #torch.optim  torch.nn  
import torch.nn as nn

def CPClayer(prediction,y_encoded):
	"""
		Should return predict_terms number of probabilities between 0 and 1
	"""

	dot_product = (y_encoded*prediction).mean(dim=-1)

	dot_product = dot_product.mean(dim=-1) 

	probabilities = torch.sigmoid(dot_product)

	return probabilities   #Really it's just one probability for each batch

class network_prediction(torch.nn.Module):
	def __init__(self,context_size,encoding_size):
		super(network_prediction,self).__init__()
		
		self.fcnet = nn.Linear(context_size,encoding_size,bias=False)
		
	def forward(self,context,predict_terms):
		"""
		forward pass of this layer
		"""
		output = []
		for i in range(predict_terms):
			output.append(self.fcnet(context))  #I'm hoping this creates a different FCnet for each
		
		
		output = torch.stack(output,dim=1)  # hoping this turns the list of torch tensors into one torch tensor
		
		
		return output

class EncoderRNN(torch.nn.Module):
	def __init__(self, input_size, hidden_size):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.gru = torch.nn.GRU(input_size, hidden_size)

	def forward(self, input, hidden):

		output, hidden = self.gru(input, hidden)
		return output, hidden

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=torch.device('cpu'))


class d1convnet(torch.nn.Module):
	def __init__(self,encoderparameters):
		"""
		encoderparameters are out channels,kernel_size
		"""
		super(d1convnet,self).__init__()
		if encoderparameters == None:
			self.net1 = torch.nn.Sequential(
				torch.nn.Conv1d(in_channels=1,out_channels=2,kernel_size=8,stride=8,padding=0),
				torch.nn.ReLU()
				)
		else:
			self.net1 = torch.nn.Sequential(
				torch.nn.Conv1d(in_channels=1,out_channels=encoderparameters[0],kernel_size=encoderparameters[1],stride=encoderparameters[1],padding=0),
				torch.nn.ReLU()
				)
	def forward(self,x):
		out = self.net1(x)
		return out


def encode_predict_terms(encoder,predict_terms,num_preds):
	"""
	Applies encoder in a timedistributed manner to the predict terms
	"""
	output = []
	for j in range(num_preds):
		output.append(encoder(predict_terms[:,j,None]))
	output = torch.stack(output,dim=1)
	
	return output






class CPC_net(torch.nn.Module):
	"""
	Takes all of the smaller nets and does a single forward pass of cpc on a given minibatch 

	3 networks, calls CPC layer calls encode predict terms

	ARparameters should be RNN_hidden_size and encoding size
	Predictparameters should be context size and encoding size
	"""
	def __init__(self,encoderparameters=None,ARparameters=None):      #It takes parameters for declaring the particular smaller NN's and declares them on init
		super(CPC_net, self).__init__()

		if ARparameters and encoderparameters==None:
			ARparameters = 512/8,16  #nspins/filter size, 16
			encoderparameters = 2,8
		elif (ARparameters == None):
			ARparameters = 512/encoderparameters[1],16  #nspins/filter size, 16
		self.ARparameters = ARparameters
		self.encoderparameters = encoderparameters
		self.encoder = d1convnet(encoderparameters) #Init the three other custom modules inside this custom module
		self.AR = EncoderRNN(ARparameters[0]*encoderparameters[0],ARparameters[1])
		self.predict = network_prediction(ARparameters[1],ARparameters[0]*encoderparameters[0])
		self.AR_output_dim = ARparameters[1]

	def forward(self,x):
		"""
		split minibatch bigtuple into X and Y before feeding it to this. Compute loss outside of this network forward pass.
		"""
		terms,predict_terms = x[0][0],x[1][0]

		batch_size = len(terms)
		num_terms = len(terms[0])
		num_preds = len(predict_terms[0])

		AR_hidden = self.AR.initHidden()

		encoderlist = []
		for j in range(num_terms):
			encoded_thing  = self.encoder(terms[:,j,None])
			encoderlist.append((self.encoder(terms[:,j,None]),terms[:,j,None]))
			
			if j<range(num_terms)[-1]: 
				AR_output,AR_hidden = self.AR(encoded_thing.view(-1,1,self.ARparameters[0]*self.encoderparameters[0]),AR_hidden)   
			else:
				context,AR_hidden = self.AR(encoded_thing.view(-1,1,self.ARparameters[0]*self.encoderparameters[0]),AR_hidden) 


		encoded_preds = encode_predict_terms(self.encoder,predict_terms,num_preds)   # This doesnt output the right sized thing
		# encoded_preds =  encoded_preds.view(encoded_preds.size()[0],encoded_preds.size()[1],-1) #collapses all but the first two dimensions of encoded_preds to allow for ambiguity of convnet   EDIT now does that when calling CPClayer below

		predictions = self.predict(context.reshape(batch_size,self.AR_output_dim),num_preds)

		probabilities = CPClayer(predictions,encoded_preds.view(encoded_preds.size()[0],encoded_preds.size()[1],-1))

		return probabilities


