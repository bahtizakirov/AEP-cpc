import numpy as np
import matplotlib.pyplot as plt
import torch


from torch.utils.data import Dataset




class special_data_loader(torch.utils.data.DataLoader):
    """
    dataloader that inherits properties of DataLoader while adding the functionality to call massage.   
    
    For future reference, add more keyword args than just dataset and shuffle.
    """
    def __init__(self,dataset,shuffle):
        super(special_data_loader,self).__init__(dataset,shuffle=True)
    def call_massage(self):
        self.dataset.massage_data()



class AEP_Dataset(Dataset):
    '''
	data_path,terms,positive,predict,batchsize,imagedim,time_span
    '''
    def __init__(self,data_path,terms,positive_samples,predict_terms,minibatch_size,image_dimension=512,time_span=50):
        """
        """
        self.time_span = time_span
        assert terms+predict_terms < time_span  #time span
        self.X = np.load(data_path)

        self.X = self.X[:,:self.time_span,:]   #Just keep T of the time series
        if data_path==('datatest3.npy'):
            self.X = self.X[:1000,:,:]

        
        
        self.terms = terms
        self.positive_samples = positive_samples
        self.predict_terms = predict_terms
        self.minibatch_size = minibatch_size
        
#         self.time_span = self.X.shape[1]
        
        
        self.massage_data()
        
    def massage_data(self):
        """
        """
        self.massaged_dataset = []
        minibatch_size = self.minibatch_size
        
        self.num_minibatches = self.X.shape[0]/self.minibatch_size
        assert self.num_minibatches%1 < 1e-7
        
        
        whole_data_idxs = range(self.X.shape[0])

        
        
        for _ in range(self.num_minibatches):
            
            this_idx = np.random.choice(whole_data_idxs,minibatch_size,replace=False)   #labels mbs pieces of 30x81x81 data            
            for i in range(self.minibatch_size):
                whole_data_idxs.remove(this_idx[i])
            
            
            
            minibatch = self.X[this_idx]
            
            
            
            
            sentence_labels = np.zeros(self.minibatch_size)
            sentence_labels[np.random.choice(np.arange(self.minibatch_size),self.positive_samples,replace=False)] = 1          #These guys become 1

            termidxs = np.empty((self.minibatch_size,self.terms))
            predictidxs = np.empty((self.minibatch_size,self.predict_terms))
            
            
            for i in range(self.minibatch_size):
                if (sentence_labels[i]):
                    #Positive Sample
                    length = self.terms + self.predict_terms
                    sentence = self.grab_n_indices_from_range(length,np.arange(self.time_span))
                    termsentence = sentence[:self.terms]
                    predictsentence = sentence[-self.predict_terms:]

                else:
                    #Negative Sample
                    length = self.terms + self.predict_terms
                    sentence = self.grab_n_indices_from_range(length,np.arange(self.time_span))
                    randoms = np.random.randint(1,self.minibatch_size,size=self.predict_terms)
                    sentence[-self.predict_terms:] = np.mod(sentence[-self.predict_terms:] + randoms ,self.minibatch_size) #corrupt predict terms


                    termsentence = sentence[:self.terms]
                    predictsentence = sentence[-self.predict_terms:]


                termidxs[i] = termsentence
                predictidxs[i] = predictsentence
            #could merge these last two loops
            termslist = []
            predslist = []

            for i in range(self.minibatch_size):   #This part is off
                terms = minibatch[i,termidxs[i].astype(int),...]
                predict_terms = minibatch[i,predictidxs[i].astype(int),...]
#                 termspreds.append([terms,predict_terms])
                termslist.append(terms)
                predslist.append(predict_terms)

            this_minibatch_tuple = ((np.array(termslist),np.array(predslist)),sentence_labels)

            self.massaged_dataset.append(this_minibatch_tuple)
            
            
            
            
            
#             self.massaged_dataset.append(this_minibatch_tuple)
    def grab_n_indices_from_range(self,length,sequence):
        """
        """
        max_start = len(sequence) - length +1
        start = np.random.randint(0,max_start)
        mask = np.arange(start,start+length)
        
        return sequence[mask]
    def __getitem__(self,idx):
        """
        """
        return self.massaged_dataset[idx]
    def __len__(self):
        """
        """
        return self.num_minibatches



if __name__ == '__main__':

	print ('Dont run this file you dummy')