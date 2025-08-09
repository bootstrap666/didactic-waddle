# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
def windowsMachine():
    return (os.name == 'nt') or (not torch.cuda.is_available())
if windowsMachine():
    import pywt
else:
    from cr.sparse import lop
    from jax import random
    import jax.numpy as jnp


class csImagecodec:
    def __init__(self,wavelet_family,save_video,weighthistory):
        self.__iswindowsmachine = windowsMachine()
        self.__wavelet_family = wavelet_family
        self.__savevideo = save_video
        self.__weighthistory = weighthistory
        if not self.__iswindowsmachine:
            seed = 777
            self.__key = random.PRNGKey(seed)
            

    def encode_image(self, img, compression_pct, wavelet_level):
        self.__compression_pct = compression_pct
        [x, self.__shape] = self.imageToVector(img)
        img_array = np.array(img)

        if self.__iswindowsmachine:
            coeffs = pywt.wavedec2(img_array, self.__wavelet_family, level=wavelet_level)
            godinez, self.__coeff_slices = pywt.coeffs_to_array(coeffs)
            self.__xw = np.reshape(godinez, np.prod(self.__shape))
            # coeffs = pywt.wavedec(np.array(x), self.__wavelet_family, level=8)
            # self.__xw, self.__coeff_slices = pywt.coeffs_to_array(coeffs)
        else:
            DWT2_op = lop.dwt2D(self.__shape, wavelet=self.__wavelet_family, level=wavelet_level)
            self.__DWT2_op = lop.jit(DWT2_op)
            coeffs = self.__DWT2_op.times(img_array)
            self.__xw = np.reshape(coeffs, np.prod(self.__shape))
            
            #self._DWT_op = lop.dwt(len(x), wavelet=self.__wavelet_family, level=8)
            #self.__xw = self._DWT_op.times(x)

        N = len(self.__xw)
        M = int(self.__compression_pct*N)

        if self.__iswindowsmachine:
            self.A = np.random.normal(loc=0.0, scale=1.0/N, size=(N,M)) # Matriz de medição
            y = np.transpose(self.A) @ self.__xw
        else:
            self.A = (1.0/jnp.sqrt(N)) * random.normal(self.__key, shape=(N,M))
            self.__key, self.__subkey = random.split(self.__key)
            y = jnp.transpose(self.A) @ self.__xw
        return y
    
    def decode_image(self,decimation,iterations,step_size,y):

        N = len(self.__xw)
        M = int(self.__compression_pct*N)
        if self.__iswindowsmachine:
              #y = np.transpose(self.A) @ xw
              B = np.linalg.inv(np.transpose(self.A) @ self.A) # economizando tempo de processamento calculando essa inversa somente uma vez
              quiescent = self.A  @ (B @ y)

              P = (np.eye(N) - self.A @ B @ np.transpose(self.A))
              xc = (P @ np.random.normal(0,1,size=(N))) + quiescent
        else: # Tentando tirar proveito de operações em JAX
              #y = jnp.transpose(self.A) @ xw
              B = jnp.linalg.inv(jnp.transpose(self.A) @ self.A)
              quiescent = self.A  @ (B @ y)
              P = jnp.subtract(jnp.eye(N), self.A @ B @ jnp.transpose(self.A))
              xc = jnp.add(quiescent, P @ random.normal(self.__subkey, shape=(N,)))
              self.__key, self.__subkey = random.split(self.__key)
        j = 0
        if (self.__weighthistory):
            self.__weights = np.zeros((N,int(iterations/decimation)))

        if self.__iswindowsmachine:
            for i in tqdm(range(iterations)):
                xc = P@(xc - step_size*np.sign(xc)) + quiescent
                if (self.__weighthistory and (not (i % decimation))):
                    self.__weights[:,j] = xc
                    j = j+1
        else:
            for i in tqdm(range(iterations)):
                xc = jnp.add(P@(jnp.subtract(xc,step_size*jnp.sign(xc))),quiescent)
                if (self.__weighthistory and (not (i % decimation))):
                    self.__weights[:,j] = xc
                    j = j+1
        
        #Gambiarra que zera os coeficientes de menor magnitude
        xc2 = np.zeros((N))
        aux = np.abs(xc)
        xc2[aux.argsort()[-M:]] = xc[aux.argsort()[-M:]]

        coeffs_rec = np.reshape(xc2, self.__shape)
        
        if windowsMachine():
            coeffs_from_arr = pywt.array_to_coeffs(coeffs_rec, self.__coeff_slices, output_format='wavedec2')
            x_rec = pywt.waverec2(coeffs_from_arr, self.__wavelet_family)
            
            # coeffs_from_arr = pywt.array_to_coeffs(xc2, self.__coeff_slices, output_format='wavedec')
            # x_rec = pywt.waverec(coeffs_from_arr, self.__wavelet_family)
        else:
            x_rec = self.__DWT2_op.trans(coeffs_rec)

            # x_rec = self._DWT_op.trans(xc2)
        imgrec = Image.fromarray(np.uint8(x_rec), 'L')
		
        #[imgrec,arrrec] = self.vectorToImage(x_rec, self.__shape)
            
        return imgrec
#plt.imshow(arrrec)
    
    def plot_weights_history(self,iterations,decimation):
        if (self.__weighthistory):
            xwvector = np.array(self.__xw)[np.newaxis]
            weighterrorvectorhistory = self.__weights- (xwvector.T @ np.ones((1, int(iterations/decimation))))
            plt.plot(weighterrorvectorhistory.T)

    def save_video(self, iterations,decimation):
        if (self.__weighthistory and self.__savevideo):
            nimages = int(iterations/decimation)
            ndigits = int(np.ceil(np.log10(nimages)))
        for i in range(nimages):
            if self.__iswindowsmachine:
                coeffs_from_arr = pywt.array_to_coeffs(self.__weights[:,i], self.__coeff_slices, output_format='wavedec2')
                x_im = pywt.waverec2(coeffs_from_arr, self.__wavelet_family)
                # coeffs_from_arr = pywt.array_to_coeffs(self.__weights[:,i], self.__coeff_slices, output_format='wavedec')
                # x_im = pywt.waverec(coeffs_from_arr, self.__wavelet_family)
            else:
                x_im = self.__DWT2_op.trans(self.__weights[:,i])
                # x_im = self.__DWT2_op.trans(self.__weights[:,i])

            filename = "img"+str(i).zfill(ndigits)+".png"
        # self.vectorToFile(x_im,self.__shape,filename)
            imgrec = Image.fromarray(np.uint8(x_im), 'L')
            imgrec.save(filename)

        os.system("ffmpeg -r 10 -i img%0"+str(ndigits)+"d.png -vcodec mpeg4 -y a.mp4")
    
#Projeção lexicográfica
    def imageToVector(self,img):
        arr = np.array(img)

        shape = arr.shape

        vector = np.reshape(arr, np.prod(shape))
        return vector, shape

    def vectorToImage(self, vector, shape):
        arr2 = np.asarray(vector.astype('uint8')).reshape(shape)

        img2 = Image.fromarray(arr2.astype(np.uint8), 'L')
        return img2, arr2

    def vectorToFile(vector,shape,filename):
        img, arr = vectorToImage(vector, shape)
        rgbimg = Image.new("RGBA", shape)
        rgbimg.paste(img)
        rgbimg.save(filename)
