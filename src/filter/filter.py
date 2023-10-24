import numpy as np

class Filter:
    def __init__(self,alpha=1e-6,iters=200):
        self.alpha=alpha
        self.iters=iters


    def gradient_descent(self,x,y,w0,b0,n):
        w=w0
        b=b0
        def derivative(x,y,w,b,n):
            dj_dw=np.dot(w*x+b-y,x)/n
            dj_db=(w*x+b-y).sum()/n
            return dj_dw,dj_db

        def cost_function(x,y,w,b,n):
            return np.dot(w*x+b-y,w*x+b-y)/(2*n)
        
        cost_arr=[cost_function(x,y,w,b,n)]
        for i in range(self.iters):
            dj_dw,dj_db=derivative(x,y,w,b,n)
            w-=self.alpha*dj_dw
            b-=self.alpha*dj_db
            cost_arr+=[cost_function(x,y,w,b,n)]
        return w,b,cost_arr

    def train(self,input_img,filtered_img):
        x=np.array(input_img,dtype='uint16').reshape(input_img.shape[0]*input_img.shape[1],3)
        y=np.array(filtered_img,dtype='uint16').reshape(filtered_img.shape[0]*filtered_img.shape[1],3)
        if(x.shape!=y.shape):
            print(f"The image dimensions dont match {x.shape} and {y.shape}")
            return
        
        self.w=np.full((3,),0,dtype='float64')
        self.b=np.full((3,),0,dtype='float64')
        self.w[0],self.b[0],h_cost_arr=self.gradient_descent(x[:,0],y[:,0],1,1,x[:,0].shape[0])
        self.w[1],self.b[1],s_cost_arr=self.gradient_descent(x[:,1],y[:,1],1,1,x[:,1].shape[0])
        self.w[2],self.b[2],v_cost_arr=self.gradient_descent(x[:,2],y[:,2],1,1,x[:,2].shape[0])
        print("Model trained succssfully")
    
    def apply_filter(self,input_img):
        input_img=np.array(input_img,dtype='uint16')
        output_img=input_img*self.w+self.b
        output_img[output_img>255]=255
        output_img[output_img<0]=0
        return np.array(output_img,dtype='uint8')
